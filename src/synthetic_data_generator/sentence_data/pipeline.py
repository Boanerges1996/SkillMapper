from langgraph.graph import StateGraph
from nodes import load_esco_node, process_all_skills_node, finish_node, State


def build_graph(llm) -> StateGraph:
    graph = StateGraph(state_schema=State)

    graph.add_node("load_esco", load_esco_node)
    graph.add_node("process_skills", lambda state: process_all_skills_node(state, llm))
    graph.add_node("finish", finish_node)

    graph.set_entry_point("load_esco")
    graph.add_edge("load_esco", "process_skills")
    graph.add_edge("process_skills", "finish")
    graph.set_finish_point("finish")

    return graph
