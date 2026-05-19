from agent.graph import compile_contract_agent
from rag.preload import preload_startup_models

preload_startup_models()
graph = compile_contract_agent(with_checkpointer=False)
