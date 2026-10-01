import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

try:
    from agent_book import graph
except ImportError:
    from basic_agent.agent_book import graph


def texto_da_resposta(content) -> str:
    if isinstance(content, list):
        return "\n\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("text")
        )
    return content


st.title("Pesquisa de livros de escalada!")

st.write(
    "Este é um agente que faz pesquisas de livros de escalada. "
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(texto_da_resposta(msg.content))

question = st.chat_input("Qual livro de escalada você busca?")

if question:
    st.session_state.messages.append(HumanMessage(content=question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Procurando livros..."):
            try:
                result = graph.invoke({"messages": st.session_state.messages})
            except Exception as exc:
                st.error(f"Erro ao executar o agente: {exc}")
                st.stop()

        response = texto_da_resposta(result["messages"][-1].content)
        st.markdown(response)

    # Guarda só pergunta/resposta finais, sem as mensagens internas das ferramentas
    st.session_state.messages.append(AIMessage(content=response))

