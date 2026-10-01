"""CGSA (SimCSE / InfoNCE) stage of KG-BiLM, data-parallel over every visible GPU.

    torchrun --standalone --nproc_per_node 2 train_cgsa_ddp.py cgsa_kaggle.json

This is `experiments/run_cgsa.py` from the KG-BiLM reference (same model, LoRA, loss, data and
arguments; the config is read by the same HfArgumentParser dataclasses) with four changes that
make multi-GPU training correct rather than just multi-process:

  1. The forward goes through the DDP-wrapped `model` that Trainer passes to compute_loss. The
     reference calls `self.model(...)`, which is the unwrapped module: DDP's reducer is never
     prepared, gradients are not all-reduced, and each GPU trains its own diverging copy.
  2. Both SimCSE views go through one forward pass (they are the same text, and dropout is drawn
     per element, so this is the same computation as two passes). One DDP forward per backward
     is the pattern DDP expects.
  3. In-batch negatives are gathered across ranks with gradient (llm2vec's HardNegativeNLLLoss
     already does this). The gather helper is re-bound to use the tensor's own device instead of
     the literal "cuda", so the same code runs under gloo on CPU for the local test.
  4. Resume from a Trainer checkpoint works: the checkpoint holds a PEFT adapter inside the
     LLM2Vec wrapper, which Trainer's own loader does not recognise.

Two runtime checks make "parallel" observable in the log rather than assumed:
  * the similarity matrix of the first step must be (B*W) x (B*W), i.e. every query sees the
    negatives from every GPU;
  * after step 1 and every `--sync_check_steps`, a fingerprint of the trainable weights is
    all-gathered and must match across ranks (a desynchronised run fails immediately).

Gradient maths: every rank computes the same loss over the gathered batch. The gather's backward
sums the gradients of all ranks into each rank's own slice (W * dL/d rep_i) and DDP then averages
over W, so the parameter update equals single-GPU training on the global batch.
"""
import json
import os
import sys
import time
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))
for candidate in (os.path.join(HERE, "experiments"), HERE):
    if os.path.exists(os.path.join(candidate, "run_cgsa.py")):
        sys.path.insert(0, candidate)
        break

import torch
import torch.distributed as dist
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import HfArgumentParser, TrainerCallback, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint

import llm2vec.loss.HardNegativeNLLLoss  # noqa: F401 (the package re-exports the class under the module's name)
from llm2vec import LLM2Vec
from llm2vec.dataset.utils import load_dataset
from llm2vec.loss.loss_utils import all_gather_with_grad, cos_sim
from llm2vec.loss.utils import load_loss

import run_cgsa as ref


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world():
    return dist.get_world_size() if dist.is_initialized() else 1


def log0(*args):
    if rank() == 0:
        print(*args, flush=True)


def gather_rows(tensor, group=None, async_op=False, mismatched_axis=0):
    """llm2vec's mismatched_sizes_all_gather with device=tensor.device (it hardcodes "cuda")."""
    sizes = [torch.zeros(1, dtype=torch.int64, device=tensor.device) for _ in range(world())]
    dist.all_gather(sizes, torch.tensor([tensor.shape[mismatched_axis]], device=tensor.device), group=group)
    sizes = torch.cat(sizes).cpu().tolist()
    shape = list(tensor.shape)
    shape[mismatched_axis] = max(sizes)
    padded = torch.zeros(shape, device=tensor.device, dtype=tensor.dtype)
    padded.narrow(mismatched_axis, 0, tensor.shape[mismatched_axis])[...] = tensor
    out = [torch.zeros_like(padded) for _ in range(world())]
    out = list(all_gather_with_grad(out, padded, group, async_op))
    return [t.narrow(mismatched_axis, 0, n) for t, n in zip(out, sizes)]


sys.modules["llm2vec.loss.HardNegativeNLLLoss"].mismatched_sizes_all_gather = gather_rows


@dataclass
class KaggleArguments:
    sync_check_steps: int = field(default=100, metadata={"help": "Check replica weights every n steps"})
    resume: str = field(default="auto", metadata={"help": "auto | none | <checkpoint dir>"})


class DDPSimCSETrainer(ref.SimCSETrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        features, labels = inputs
        q, d = features[0], features[1]
        same_shape = len(features) == 2 and all(
            (q[k] is None and d[k] is None) or (q[k] is not None and d[k] is not None and q[k].shape == d[k].shape)
            for k in q
        )
        if same_shape:
            both = {k: None if q[k] is None else torch.cat([q[k], d[k]]) for k in q}
            q_reps, d_reps = model(both).chunk(2)
            loss = self.loss_function(q_reps, d_reps, None)
        else:  # hard-negative datasets or differently padded views: the reference path, through DDP
            reps = [model(f) for f in features]
            loss = self.loss_function(reps[0], reps[1], reps[2] if len(reps) > 2 else None)
        return (loss, None) if return_outputs else loss

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        state = load_file(os.path.join(resume_from_checkpoint, "adapter_model.safetensors"))
        result = set_peft_model_state_dict(self.model.model, state)
        if getattr(result, "unexpected_keys", None):
            raise RuntimeError(f"unexpected adapter keys in {resume_from_checkpoint}: {result.unexpected_keys[:5]}")
        log0(f"resumed adapter weights from {resume_from_checkpoint}")

    def _load_rng_state(self, checkpoint):
        # transformers 4.44 predates torch 2.6's weights_only=True default; rng_state_*.pth holds numpy
        # and python RNG state, which that mode refuses. These are checkpoints this script wrote.
        load = torch.load
        torch.load = lambda *a, **k: load(*a, **{**k, "weights_only": False})
        try:
            super()._load_rng_state(checkpoint)
        finally:
            torch.load = load


class NegativesProbe:
    """Wraps the loss similarity to record the shape of the first similarity matrix."""

    def __init__(self, expected):
        self.expected, self.shape = expected, None

    def __call__(self, a, b):
        if self.shape is None:
            self.shape = (a.shape[0], b.shape[0])
            log0(f"[parallel] similarity matrix {self.shape[0]} x {self.shape[1]} "
                 f"(per-device batch {a.shape[0] // world()} x {world()} ranks)")
            if self.shape != (self.expected, self.expected):
                raise RuntimeError(f"expected {self.expected} x {self.expected} gathered negatives, got {self.shape}")
        return cos_sim(a, b)


class ReplicaSyncCheck(TrainerCallback):
    def __init__(self, params, every):
        self.params, self.every, self.checks = params, every, []

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if world() == 1 or not (step == 1 or step % self.every == 0):
            return
        with torch.no_grad():
            flat = torch.cat([p.detach().float().flatten() for p in self.params])
            fp = torch.stack([flat.sum(), flat.pow(2).sum(), flat[::997].sum()])
            gathered = [torch.zeros_like(fp) for _ in range(world())]
            dist.all_gather(gathered, fp)
            diff = max((g - gathered[0]).abs().max().item() for g in gathered)
        scale = gathered[0].abs().max().item() + 1e-12
        self.checks.append({"step": step, "max_abs_diff": diff, "rel": diff / scale})
        log0(f"[parallel] step {step}: replica weight fingerprint max diff {diff:.3e} (rel {diff / scale:.1e})")
        if diff / scale > 1e-6:
            raise RuntimeError(f"replicas diverged at step {step}: DDP is not synchronising gradients")


def main():
    parser = HfArgumentParser(
        (ref.ModelArguments, ref.DataTrainingArguments, TrainingArguments, ref.CustomArguments, KaggleArguments)
    )
    model_args, data_args, training_args, custom_args, kaggle_args = parser.parse_json_file(os.path.abspath(sys.argv[1]))
    set_seed(training_args.seed)
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    log0(f"[parallel] world size {world()}, per-device batch {training_args.per_device_train_batch_size}, "
         f"global batch {training_args.per_device_train_batch_size * world() * training_args.gradient_accumulation_steps}, "
         f"backend {dist.get_backend() if dist.is_initialized() else 'none (single process)'}")

    dataset = load_dataset(data_args.dataset_name, split="train", file_path=data_args.dataset_file_path)
    n = len(dataset) if data_args.max_train_samples is None else min(len(dataset), data_args.max_train_samples)
    train_examples = [dataset[i] for i in range(n)]
    log0(f"{n} training sentences")

    torch_dtype = model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    model = LLM2Vec.from_pretrained(
        base_model_name_or_path=model_args.model_name_or_path,
        enable_bidirectional=model_args.bidirectional,
        peft_model_name_or_path=model_args.peft_model_name_or_path,
        merge_peft=True,
        pooling_mode=model_args.pooling_mode,
        max_length=model_args.max_seq_length,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
        attention_dropout=custom_args.simcse_dropout,
    )
    model.model = ref.initialize_peft(
        model.model, lora_r=custom_args.lora_r, lora_alpha=2 * custom_args.lora_r, lora_dropout=custom_args.lora_dropout
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    if training_args.fp16 and any(p.dtype != torch.float32 for p in trainable):
        raise ValueError("fp16 mixed precision needs fp32 trainable weights: set torch_dtype to float32")

    loss = load_loss(custom_args.loss_class, scale=custom_args.loss_scale)
    probe = NegativesProbe(training_args.per_device_train_batch_size * world())
    loss.similarity_fct = probe
    sync = ReplicaSyncCheck(trainable, kaggle_args.sync_check_steps)

    trainer = DDPSimCSETrainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        data_collator=ref.DefaultCollator(model),
        tokenizer=model.tokenizer,
        loss_function=loss,
        callbacks=[ref.StopTrainingCallback(custom_args.stop_after_n_steps), sync],
    )

    resume = None
    if kaggle_args.resume == "auto" and os.path.isdir(training_args.output_dir):
        resume = get_last_checkpoint(training_args.output_dir)
    elif kaggle_args.resume not in ("auto", "none", ""):
        resume = kaggle_args.resume
    log0(f"resume from: {resume or 'scratch'}")

    start = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(training_args.output_dir)  # final adapter at the top level (the reference only kept checkpoints)

    if trainer.is_world_process_zero():
        summary = {
            "world_size": world(),
            "per_device_batch": training_args.per_device_train_batch_size,
            "global_batch": training_args.per_device_train_batch_size * world() * training_args.gradient_accumulation_steps,
            "similarity_matrix": probe.shape,
            "replica_sync_checks": sync.checks,
            "global_step": trainer.state.global_step,
            "train_loss": result.training_loss,
            "wall_seconds": round(time.time() - start, 1),
            "resumed_from": resume,
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "log_history": trainer.state.log_history,
        }
        with open(os.path.join(training_args.output_dir, "train_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        log0(json.dumps({k: v for k, v in summary.items() if k != "log_history"}, indent=2))

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
