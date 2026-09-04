import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
)
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

load_dotenv()


# ============================================================
# API KEYS
# ============================================================

HF_TOKEN = os.environ.get("HF_TOKEN")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")


# ============================================================
# TAVILY — procura livros na web
# ============================================================

tavily_client = (
    TavilySearch(
        max_results=5,
        topic="general",
        tavily_api_key=TAVILY_API_KEY,
    )
)


# ============================================================
#  Procura do livro na web
# ============================================================
@tool
def web_search(query:str) -> str:
    """Busque na web somente por livros de escalada! Caso não encontre, diga que não encontrou.
        Se o usuário pedir sobre outros temas , retorne que o agente é especializado em
        livros de escalada e não pode ajudar com outros temas.
    """
    try:
        results = tavily_client.invoke({"query": query})
    except Exception as exc:
        return f"Erro ao consultar a web: {exc}"

    items = results.get("results", [])

    if not items:
        return f"No results found for '{query}'"

    resposta = []
    for item in items:
        resposta.append(
            f"Title: {item.get('title')}\n"
            f"URL: {item.get('url')}\n"
            f"Description: {item.get('content','')[:400]}"
        )
    return "\n\n".join(resposta)
# ============================================================
# LLM - Hugging face
# ============================================================

llm = (
    HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        task="text-generation",
        max_new_tokens=700,
        huggingfacehub_api_token=HF_TOKEN,
    )
)

chat_model = ChatHuggingFace(llm=llm)
llm_with_tools = chat_model.bind_tools([web_search])


# ============================================================
# criação dos nodes
# ============================================================

class State(TypedDict):
    question: str
    answer: str
    messages: Annotated[list, add_messages]

def agent_node(state: State):
    prompt = f"Pergunta do usuário: {state['question']}"

    if llm_with_tools is None:
        return {
            "answer": "O agente não está configurado corretamente. Verifique as chaves da API do Hugging Face e Tavily.",
            "messages": [],
        }

    messages = state.get("messages", [])
    if not messages:
        messages = [HumanMessage(content=prompt)]

    try:
        response = llm_with_tools.invoke(messages)
    except Exception as exc:
        return {
            "answer": f"Não foi possível executar o agente neste ambiente: {exc}",
            "messages": [],
        }

    return {"answer": response.content, "messages": [response]}

# ============================================================
# LANGGRAPH
# ============================================================


graph_builder = StateGraph(State)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode([web_search]))
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges(
    "agent",
    tools_condition,
)
graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile()
