"""Collect the trained adapters into models/ and write MANIFEST.json (sha256 of every file).

    python scripts/export_weights.py --from /path/to/kev/runs/kev_graph

Source run directories (the evaluated checkpoints):
  techdoc-mntp/          -> base-adapters/mntp      (MNTP, KG-BiLM stage 1)
  techdoc-cgsa/          -> base-adapters/cgsa      (CGSA, KG-BiLM stage 2)
  sup-B/final/           -> defrost-ret-b               (supervised retrieval, arm B)
  kevrerank/final/       -> defrost-rerank (+ head.pt)  (listwise cross-encoder)"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP = {"techdoc-mntp": "base-adapters/mntp", "techdoc-cgsa": "base-adapters/cgsa", "sup-B/final": "defrost-ret-b",
       "kevrerank/final": "defrost-rerank"}
FILES = ["adapter_config.json", "adapter_model.safetensors", "head.pt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--to", default=str(ROOT / "models"))
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.to)
    files = {}
    for s, d in MAP.items():
        (dst / d).mkdir(parents=True, exist_ok=True)
        for f in FILES:
            p = src / s / f
            if p.exists():
                shutil.copy2(p, dst / d / f)
                data = (dst / d / f).read_bytes()
                files[f"{d}/{f}"] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    manifest = {"version": "1.0.0", "base_model": "Qwen/Qwen2.5-0.5B",
                "base_revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
                "chain": {"defrost-ret-b": ["base-adapters/mntp", "base-adapters/cgsa", "defrost-ret-b"],
                          "defrost-rerank": ["base-adapters/mntp", "base-adapters/cgsa", "defrost-rerank", "defrost-rerank/head.pt"]},
                "files": files}
    (dst / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(files)} files, {sum(v['bytes'] for v in files.values()) / 1e6:.0f} MB -> {dst}")


if __name__ == "__main__":
    main()
