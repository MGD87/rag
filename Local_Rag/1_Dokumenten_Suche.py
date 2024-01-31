from local_rag import LocalRag
import os
import yaml
import streamlit as st

st.set_page_config(page_title="Dokumentensuche")
st.title("Dokumentensuche")
st.subheader("Durchsuche deine Dokumente um Antworten zu finden.")
st.write("Deine Dokumentenablagen ist ein Aggregat aller Dokumente.")
if os.path.isfile("config_real.yaml"):
    config_file = "config_real.yaml"
else:
    config_file = "config.yaml"

required_keys = ["database_name", "user", "password", "host", "db_port", "embedding_batches",
                 "model_name", "ollama_api_url", "temperature"]


def check_yaml_population(yaml_file, required_keys):
    try:
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        if not data:
            return False

        for key in required_keys:
            if key not in data or data[key] is None:
                return False

            elif data[key] == '':
                return False

        return True
    except Exception as e:
        print(f"Error checking YAML file: {e}")
        return False


if check_yaml_population(config_file, required_keys):
    rag_class = LocalRag(config_file)

    # Get the documents on file
    document_list = rag_class.get_sql_documents()
    
    if document_list != []:
        query = st.chat_input("Was möchtest du wissen?")

        pdf_mapping = {entry[1]: entry[2] for entry in document_list}

        # Create a dropdown menu
        selected_pdf_name = st.selectbox("Wähle ein Dokument aus", options=list(pdf_mapping.keys()))

        # Display the selected PDF ID
        selected_pdf_id = pdf_mapping[selected_pdf_name]

        k = st.slider("Quellen zur Wiedergabe.", 1, 10, 5)

        # Chunking option for uploading document
        reranking_option = {
            'Kein Reranking': 'no',
            'Reranking': 'rerank'
        }
        rank_option = st.radio("Reranking Methode:", options=reranking_option)
        rank_strategy = reranking_option[rank_option]

        if query:
            st.write(f"User: {query}")

            with st.status("Answering query...", expanded=True) as status:
                # Reranking strategy and setting K depending on it
                st.write("Suche nach Resultaten...")
                if rank_strategy == "rerank":
                    paragraph_id_list, cosine_results = rag_class.retrieve_documents(query, k * 4, selected_pdf_id)
                else:
                    paragraph_id_list, cosine_results = rag_class.retrieve_documents(query, k, selected_pdf_id)

                # Getting sources
                st.write("Suche nach Quellen...")
                sources_list = rag_class.get_paragraph_sources(paragraph_id_list)

                # Reranking strategy
                if rank_strategy == "rerank":
                    st.write("Reranking der Quellen...")
                    sorted_sources = rag_class.rerank_sources(query, sources_list)
                    sorted_sources = sorted_sources[:k]
                    context_for_llm = "".join(sorted_sources)
                else:
                    sorted_sources = sources_list
                    context_for_llm = "".join(sources_list)

                # Send to LLM
                st.write("Verarbeitung durch LLM...")
                answer = rag_class.make_llm_request(query, context_for_llm)
                status.update(label="Antwort gefunden!", state="complete", expanded=False)

            with st.chat_message("assistant"):
                st.write(answer)

            for idx, item in enumerate(sorted_sources):
                with st.expander(f"Quelle {idx + 1}"):
                    st.markdown(item)

    else:
        st.markdown("Bitte Lade dokumente hoch.")

else:
    st.write("Deine Einstellungen sind nicht korrekt ausgefüllt.")
