import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
)
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, StateGraph


# ============================================================
# API KEYS
# ============================================================

load_dotenv()


HF_TOKEN = os.environ.get("HF_TOKEN")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")


# ============================================================
# LLM — HUGGING FACE
# ============================================================

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=700,
    huggingfacehub_api_token=HF_TOKEN,
)

chat_model = ChatHuggingFace(
    llm=llm
)


# ============================================================
# TAVILY — procura livros na web
# ============================================================

web_search = TavilySearch(
    max_results=5,
    topic="general",
    tavily_api_key=TAVILY_API_KEY,
)


# ============================================================
# criação dos nodes
# ============================================================

class State(TypedDict):
    question: str
    search_results: list
    answer: str


# ============================================================
#  Procura do livro na web
# ============================================================

def search_web(state: State):
    results = web_search.invoke({'query': state['question']})
    return {'search_results': results.get('results', [])}


# ============================================================
# Resposta
# ============================================================

def generate_answer(state: State):

    results = state["search_results"]

    if not results:

        return {
            "answer": (
                "Não encontrei resultados na web para essa pesquisa."
            )
        }

    context = "\n\n".join([f"""TITLE: {result.get("title", "")}URL: {result.get("url", "")}
CONTENT: {result.get("content", "")}
"""
            for result in results
        ]
    )

    prompt = f"""
            Você é um agente especializado em pesquisar
            livros de escalada na internet.

            Pergunta do usuário:

            {state["question"]}

            Resultados encontrados na internet:

            {context}

            Sua tarefa:

            1. Identifique quais resultados realmente
            mencionam livros de escalada.

            2. Para cada livro informe em formato de tópico:
            - título
            - autor, se disponível
            - resumo do conteudo em uma linha
            - fonte/URL

            3. Se o resultado não for um livro, não traga.

            4. Se não houver livros relevantes, responda que não encontrou livros de escalada.
            5.Não invente resultados, se não encontrar traga como missing.

            Responda em português.
            """

    response = chat_model.invoke(prompt)

    return {
        "answer": response.content
    }


# ============================================================
# LANGGRAPH
# ============================================================

graph_builder = StateGraph(State)


graph_builder.add_node(
    "search_web",
    search_web
)

graph_builder.add_node(
    "generate_answer",
    generate_answer
)


# ============================================================
# EDGES
# ============================================================

graph_builder.add_edge(
    START,
    "search_web"
)

graph_builder.add_edge(
    "search_web",
    "generate_answer"
)

graph_builder.add_edge(
    "generate_answer",
    END
)


graph = graph_builder.compile()