from local_rag import LocalRag
import streamlit as st
import yaml
import os


# Temp storage of document file
def save_uploaded_file(directory, file):
    if not os.path.exists(directory):
        os.makedirs(directory)
    full_path = os.path.join(directory, file.name)
    with open(full_path, "wb") as f:
        f.write(file.getbuffer())


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


# Page setup
st.set_page_config(page_title="Documenten Manager")
st.title("Documenten Manager")
st.subheader("Documentenupload")
st.write("CErstellen Sie einen durchsuchbaren Dokumentenspeicher, indem Sie Ihre eigenen txt-, pdf- oder docx-Dokumente hinzufügen. Sie können mehrere Dokumente in diesem durchsuchbaren Speicher ablegen.")

if check_yaml_population(config_file, required_keys):
    rag_class = LocalRag(config_file)
    # Form setup
    name_in_db = st.text_input("Dokumentenname")
    uploaded_file = st.file_uploader("Wähle eine Datei zum hinzufügen aus", key="new_vid", type=["pdf", "txt", "docx"])

    # Get the documents on file
    document_list = rag_class.get_sql_documents()

    # Chunking option for uploading document
    chunking_options = {
        'Einfach': 'simple',
        'Klein bis Groß': 'smalltobig'
    }
    chunk_option = st.radio("Strategie zur Stückelung von Dokumenten:", options=chunking_options)
    chunk_strategy = chunking_options[chunk_option]
    upload_file = st.button("Embed Document")


    # Upload a new document
    if upload_file and name_in_db != "":
        if uploaded_file is not None:
            # Save the file to a folder
            current_working_directory = os.getcwd()
            file_directory = os.path.join(current_working_directory, "temp_doc_storage")

            with st.status("Dokument hochladen...", expanded=True) as status:
                st.write("Datei hochladen...")
                save_uploaded_file(file_directory, uploaded_file)

                st.write("Zerteilen des Dokuments...")
                if uploaded_file.type == "application/pdf":
                    list_to_embed, paragraph_keys, doc_id = rag_class.pdf_document_reader(uploaded_file.name, name_in_db, chunk_strategy=chunk_strategy)

                elif uploaded_file.type == "text/plain":
                    list_to_embed, paragraph_keys, doc_id = rag_class.txt_document_reader(uploaded_file.name, name_in_db, chunk_strategy=chunk_strategy)

                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    list_to_embed, paragraph_keys, doc_id = rag_class.docx_document_reader(uploaded_file.name, name_in_db, chunk_strategy=chunk_strategy)

                st.write("Generierung von Vectoren...")
                embeddings = rag_class.create_batch_embeddings(list_to_embed, doc_id)

                st.write("Hinzufügen von Vectoren in in eine Vektoren Datenbank...")
                rag_class.load_documents_db(embeddings, paragraph_keys, doc_id)

                st.write("Aufräumen...")
                os.remove(os.path.join(file_directory, uploaded_file.name))

                status.update(label="Upload completed!", state="complete", expanded=False)


    # st.divider()
    # st.subheader("Upload YouTube Video")
    # st.write("Create a searchable storage for the transcript of a YouTube video. You can multiple documents to this searchable storage.")
    # video_in_db = st.text_input("Document name", key="id_in")
    # youtube_link = st.text_input("YouTube ID")
    # chunk_option = st.radio("Video chunking strategy:", options=chunking_options)
    # chunk_strategy = chunking_options[chunk_option]
    # embed_video = st.button("Embed YouTube Video")

    # if embed_video and video_in_db != "":
    #     if youtube_link is not None:
    #         with st.status("Uploading document...", expanded=True) as status:
    #             st.write("Uploading transcript...")
    #             list_to_embed, paragraph_keys, doc_id = rag_class.youtube_reader(youtube_link, video_in_db, chunk_strategy=chunk_strategy)

    #             st.write("Generating vectors...")
    #             embeddings = rag_class.create_batch_embeddings(list_to_embed, doc_id)

    #             st.write("Adding vectors into vector database...")
    #             rag_class.load_documents_db(embeddings, paragraph_keys, doc_id)

    #             status.update(label="Upload completed!", state="complete", expanded=False)


    st.divider()
    st.subheader("Zusätzliches Dokument hinzufügen")
    st.write("Fügen Sie zusätzliche txt-, pdf- oder docx-Dokumente zu Ihrem Dokumentenspeicher hinzu. Sie können dies zu jedem Dokumentenspeicher hinzufügen, den Sie haben.")

    # Making sure there are documents to add to
    if document_list != []:
        # Mapping for selecting a document name and id to add the text to with a dropdown
        add_pdf_mapping = {entry[1]: [entry[2], entry[3]] for entry in document_list}

        # Form setup
        add_document_name = st.selectbox("Hinzufügen zum Dokument", options=list(add_pdf_mapping.keys()))
        add_pdf_id = add_pdf_mapping[add_document_name][0]
        add_chunk_strategy = add_pdf_mapping[add_document_name][1]
        add_uploaded_file = st.file_uploader("Wählen Sie eine Datei zum Hinzufügen", type=["pdf", "txt", "docx"])
        add_upload_file = st.button("Dokument hinzufügen")

        # Algorithm to add a document
        if add_upload_file:
            if add_uploaded_file is not None:
                # Save the file to a folder
                current_working_directory = os.getcwd()
                file_directory = os.path.join(current_working_directory, "temp_doc_storage")

                with st.status("Dokument hinzufügen...", expanded=True) as status:
                    st.write("Datei hochladen...")
                    save_uploaded_file(file_directory, add_uploaded_file)

                    st.write("Zerteilen eines Dokuments...")
                    if add_uploaded_file.type == "application/pdf":
                        list_to_embed, paragraph_keys, _ = rag_class.pdf_document_reader(add_uploaded_file.name, add_document_name, chunk_strategy=add_chunk_strategy, add_to_doc=True)

                    elif add_uploaded_file.type == "text/plain":
                        list_to_embed, paragraph_keys, _ = rag_class.txt_document_reader(add_uploaded_file.name, name_in_db, chunk_strategy=add_chunk_strategy, add_to_doc=True)

                    elif add_uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        list_to_embed, paragraph_keys, _ = rag_class.docx_document_reader(add_uploaded_file.name, name_in_db, chunk_strategy=add_chunk_strategy, add_to_doc=True)

                    st.write("Generierung von Vektoren...")
                    embeddings = rag_class.create_batch_embeddings(list_to_embed, add_pdf_id)

                    st.write("Hinzufügen von Vektoren zur Vektordatenbank...")
                    rag_class.load_documents_db(embeddings, paragraph_keys, add_pdf_id)

                    st.write("Aufräumen...")
                    os.remove(os.path.join(file_directory, add_uploaded_file.name))

                    status.update(label="Hochladen erfolgreich", state="complete", expanded=False)

    else:
        st.markdown("Bitte füge zuerst Dokumente hinzu.")


    # st.divider()
    # st.subheader("Add Additional YouTube Video")
    # st.write("Add additional YouTube Video to your document store. You can add this to any document store you have, including to document stores.")

    # # Making sure there are documents to add to
    # if document_list != []:
    #     # Mapping for selecting a document name and id to add the text to with a dropdown
    #     add_pdf_mapping = {entry[2]: [entry[3], entry[4]] for entry in document_list}

    #     # Form setup
    #     add_youtube_name = st.selectbox("Add to document", key="yt_name", options=list(add_pdf_mapping.keys()))
    #     add_video_id = add_pdf_mapping[add_youtube_name][0]
    #     add_chunk_strategy = add_pdf_mapping[add_youtube_name][1]
    #     add_youtube_link = st.text_input("YouTube ID", key="add_link")
    #     add_youtube_file = st.button("Add YouTube Video")

    #     # Algorithm to add a document
    #     if add_youtube_file:
    #         if add_youtube_link is not None:
    #             with st.status("Uploading document...", expanded=True) as status:
    #                 st.write("Uploading transcript...")
    #                 list_to_embed, paragraph_keys, _ = rag_class.youtube_reader(add_youtube_link, add_youtube_name, chunk_strategy=add_chunk_strategy, add_to_doc=True)

    #                 st.write("Generating vectors...")
    #                 embeddings = rag_class.create_batch_embeddings(list_to_embed, add_video_id)

    #                 st.write("Adding vectors into vector database...")
    #                 rag_class.load_documents_db(embeddings, paragraph_keys, add_video_id)

    #                 status.update(label="Upload completed!", state="complete", expanded=False)

    # else:
    #     st.markdown("First add documents.")

    st.divider()

    st.subheader("Dokument löschen")
    st.write("Löschen Sie einen Dokumentenspeicher, dies kann nicht rückgängig gemacht werden.")
    if document_list != []:
        # Create a dropdown menu for documents to display with id
        del_pdf_mapping = {entry[1]: entry[2] for entry in document_list}

        # Making the form
        del_pdf_name = st.selectbox("Wähle ein Dokument aus", options=list(del_pdf_mapping.keys()))
        del_pdf_id = del_pdf_mapping[del_pdf_name]
        delete_docs = st.button("Lösche")

        # Deleting documents
        if delete_docs:
            with st.status("Löschen eines Dokuments...", expanded=True) as status:
                st.write("Löschen eines Dokuments...")
                rag_class.delete_docs(del_pdf_id)
                status.update(label="Document entfernt!", state="complete", expanded=False)

    else:
        st.markdown("Keine Dokumente hochgeladen.")

else:
    st.write("Deine Einstellungen sind nicht korrekt ausgefüllt")
