"""
secret_agent_man/agent.py
==========================
Entry point for the smolagents CodeAgent with cascading multi-provider LLM
and code review / web tools.
Run:
    uv run python3 -m secret_agent_man.agent
"""
from secret_agent_man.cascading_model import CascadingModel
from secret_agent_man.tools import SecondOpinionTool, ReadFileTool, SearchCodeTool, RunCommandTool
from smolagents import CodeAgent
from smolagents.default_tools import DuckDuckGoSearchTool, VisitWebpageTool


def build_agent(max_tokens: int = 2048, temperature: float = 0.1) -> CodeAgent:
    """Construct and return a ready-to-run CodeAgent."""
    model = CascadingModel(max_tokens=max_tokens, temperature=temperature)
    tools = [
        SecondOpinionTool(model),
        ReadFileTool(),
        SearchCodeTool(),
        RunCommandTool(),
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
    ]
    return CodeAgent(
        tools=tools,
        model=model,
        add_base_tools=True,
        additional_authorized_imports=["os", "shutil", "pathlib"],
    )


if __name__ == "__main__":
    agent = build_agent()
    print("🤖 Agent online. Type 'exit' to quit, 'reset' to start a new conversation.")
    print("💡 Try: 'review the code in src/rag_pipeline/core/paths.py'")
    first_run = True
    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "reset":
            first_run = True
            print("🔄 Conversation reset.")
            continue
        try:
            agent.run(user_input, reset=first_run)
            first_run = False
        except Exception as e:
            print(f"⚠️  Error: {e}")
