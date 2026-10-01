"""Claude-CLI helpers for the e2e answer + judge steps (no API key needed; uses a logged-in `claude` CLI)."""
import subprocess
import time

NV = ["nv_accuracy", "nv_context_relevance", "nv_response_groundedness"]


def claude(prompt, model):
    cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "text", "--tools", "",
           "--no-session-persistence", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}']
    for _ in range(3):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
        time.sleep(20)
    return None


def claude_llm(model="haiku"):
    """LangChain LLM backed by the claude CLI, wrapped for RAGAS. Empty output raises, so RAGAS records NaN
    instead of silently scoring an empty verdict."""
    from langchain_core.language_models.llms import LLM
    from ragas.llms import LangchainLLMWrapper

    class ClaudeCLI(LLM):
        model_name: str = model

        @property
        def _llm_type(self):
            return "claude-cli"

        def _call(self, prompt, stop=None, run_manager=None, **kw):
            cmd = ["claude", "-p", prompt, "--model", self.model_name, "--output-format", "text", "--tools", "",
                   "--no-session-persistence", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}']
            for attempt in range(4):
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL)
                    if r.returncode == 0 and r.stdout.strip():
                        return r.stdout.strip()
                except subprocess.TimeoutExpired:
                    pass
                time.sleep(30 * (attempt + 1))
            raise RuntimeError("claude CLI returned no output")

    return LangchainLLMWrapper(ClaudeCLI())
