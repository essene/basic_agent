import streamlit as st

from agent_book import graph


st.title("Pesquisa de livros de escalada!")

st.write(
    "Este é um agente que faz pesquisas de livros de escalada, faça sua pergunta:"
)


with st.form("book_form"):

    question = st.text_area(
        "Qual livro de escalada você busca?"
    )

    submitted = st.form_submit_button(
        "Search"
    )


if submitted:

    if not question.strip():

        st.warning(
            "Digite uma pergunta:"
        )

    else:

        with st.spinner(
            "Procurando na web..."
        ):

            result = graph.invoke({
                "question": question,
                "search_results": [],
                "answer": ""
            })

        st.subheader(
            "Recomendações!"
        )

        st.markdown(
            result["answer"]
        )