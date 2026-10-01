# ============================================================
# Chatbot de livros de escalada — RAG (Retrieval-Augmented Generation)
#
# Como funciona:
#   1. RECUPERAR: procura na base de livros já pesquisados os
#      trechos mais parecidos com a pergunta (busca por embeddings).
#   2. AVALIAR: o LLM confere se esses trechos realmente respondem
#      à pergunta (texto "parecido" nem sempre é texto "útil").
#   3. Se não servirem, BUSCA NA INTERNET, salva os resultados
#      na base (indexação) e usa eles.
#   4. GERAR: o LLM recebe a pergunta + os trechos recuperados
#      (o "contexto") e responde usando SÓ esse contexto.
#
#   START -> recuperar -> avaliar --(serve)-------------------> gerar -> END
#                                \--(não serve)-> buscar_internet -/
# ============================================================

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEndpointEmbeddings,
)
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, MessagesState, StateGraph


# ------------------------------------------------------------
# PASSO 1 — Chaves de API (lidas do arquivo .env)
# ------------------------------------------------------------
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")


# ------------------------------------------------------------
# PASSO 2 — Base de conhecimento (vector store)
#
# Embeddings transformam um texto em números que representam
# o SIGNIFICADO dele. Assim, "como começar a escalar" encontra
# um livro "Guia para iniciantes", mesmo sem palavras iguais.
# ------------------------------------------------------------
ARQUIVO_LIVROS = Path(__file__).parent / "livros_pesquisados.json"

# De 0 a 1: quanto um trecho salvo precisa ser parecido com a pergunta
SIMILARIDADE_MINIMA = 0.6

# Quantos trechos no máximo vão para o contexto do LLM
QTD_TRECHOS = 5

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    huggingfacehub_api_token=HF_TOKEN,
)

# Se já existe o arquivo, carrega os livros salvos. Senão, começa vazio.
if ARQUIVO_LIVROS.exists():
    livros_salvos = InMemoryVectorStore.load(str(ARQUIVO_LIVROS), embeddings)
else:
    livros_salvos = InMemoryVectorStore(embeddings)


# ------------------------------------------------------------
# PASSO 3 — Busca na internet e LLM
# ------------------------------------------------------------
busca_internet = TavilySearch(max_results=5, tavily_api_key=TAVILY_API_KEY)

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=2000,
    huggingfacehub_api_token=HF_TOKEN,
)
chat_model = ChatHuggingFace(llm=llm)


# ------------------------------------------------------------
# PASSO 4 — Estado do grafo
#
# MessagesState já traz a lista "messages" (o histórico do chat).
# Acrescentamos os campos que o RAG passa de um passo para outro.
# ------------------------------------------------------------
class EstadoRAG(MessagesState):
    documentos: list[Document]  # trechos recuperados (o contexto)
    fonte: str                  # "livros já pesquisados" ou "internet"
    contexto_serve: bool        # resultado da avaliação do LLM


def ultima_pergunta(state: EstadoRAG) -> str:
    """Pega o texto da última mensagem do usuário."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


# ------------------------------------------------------------
# PASSO 5 — Nós do grafo
# ------------------------------------------------------------
def recuperar(state: EstadoRAG):
    """R do RAG: busca na base os trechos parecidos com a pergunta."""
    pergunta = ultima_pergunta(state)
    resultados = livros_salvos.similarity_search_with_score(pergunta, k=QTD_TRECHOS)

    documentos = [doc for doc, similaridade in resultados if similaridade >= SIMILARIDADE_MINIMA]
    return {"documentos": documentos, "fonte": "livros já pesquisados"}


def avaliar(state: EstadoRAG):
    """Pergunta ao LLM se os trechos recuperados respondem à pergunta."""
    documentos = state["documentos"]
    if not documentos:
        return {"contexto_serve": False}

    trechos = "\n\n".join(doc.page_content for doc in documentos)
    prompt = (
        "Os trechos abaixo mencionam livros que respondem à pergunta do usuário? "
        "Responda apenas SIM ou NAO.\n\n"
        f"PERGUNTA: {ultima_pergunta(state)}\n\nTRECHOS:\n{trechos}"
    )
    resposta = chat_model.invoke(prompt).text.strip().upper()
    return {"contexto_serve": resposta.startswith("SIM")}


def decidir_proximo_passo(state: EstadoRAG) -> str:
    """Se o contexto da base serve, vai direto gerar. Senão, busca na internet."""
    if state["contexto_serve"]:
        return "gerar"
    return "buscar_internet"


def buscar_internet(state: EstadoRAG):
    """Busca na internet e INDEXA os resultados na base para as próximas perguntas."""
    pergunta = ultima_pergunta(state)
    resposta = busca_internet.invoke({"query": pergunta})

    documentos = []
    for item in resposta.get("results", []):
        # page_content: o texto que vira embedding (título + resumo)
        # metadata: informações extras guardadas junto com o trecho
        documentos.append(
            Document(
                page_content=f"{item['title']}\n{item['content'][:1000]}",
                metadata={"titulo": item["title"], "url": item["url"]},
            )
        )

    if documentos:
        # O link é usado como id para não salvar o mesmo livro duas vezes
        links = [doc.metadata["url"] for doc in documentos]
        livros_salvos.add_documents(documentos, ids=links)
        livros_salvos.dump(str(ARQUIVO_LIVROS))

    return {"documentos": documentos, "fonte": "internet"}


INSTRUCOES = (
    "Você é um assistente especializado em livros de escalada. "
    "Responda usando SOMENTE as informações do CONTEXTO abaixo. "
    "Se o contexto não tiver livros relevantes, diga isso claramente. "
    "Não invente título, autor, preço ou link. "
    "Responda em português com no máximo 5 livros em uma tabela com "
    "título, descrição curta e link, e diga de onde vieram os livros."
)


def gerar(state: EstadoRAG):
    """A + G do RAG: monta o prompt com o contexto e gera a resposta."""
    documentos = state["documentos"]

    if not documentos:
        return {"messages": [AIMessage(content="Não encontrei livros para essa pesquisa.")]}

    # Augmented: junta os trechos recuperados em um único texto de contexto
    contexto = "\n\n".join(
        f"[{i}] {doc.page_content}\nLink: {doc.metadata['url']}"
        for i, doc in enumerate(documentos, start=1)
    )
    sistema = SystemMessage(
        content=f"{INSTRUCOES}\n\nFONTE: {state['fonte']}\n\nCONTEXTO:\n{contexto}"
    )

    # Generation: o histórico vai junto para o LLM entender perguntas de acompanhamento
    resposta = chat_model.invoke([sistema, *state["messages"]])
    return {"messages": [resposta]}


# ------------------------------------------------------------
# PASSO 6 — Montando o grafo
# ------------------------------------------------------------
builder = StateGraph(EstadoRAG)

builder.add_node("recuperar", recuperar)
builder.add_node("avaliar", avaliar)
builder.add_node("buscar_internet", buscar_internet)
builder.add_node("gerar", gerar)

builder.add_edge(START, "recuperar")
builder.add_edge("recuperar", "avaliar")
builder.add_conditional_edges("avaliar", decidir_proximo_passo, ["gerar", "buscar_internet"])
builder.add_edge("buscar_internet", "gerar")
builder.add_edge("gerar", END)

graph = builder.compile()


def listar_livros_salvos() -> list[dict]:
    """Retorna título e link de todos os livros já salvos na base."""
    return [item["metadata"] for item in livros_salvos.store.values()]
