import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
import tempfile
import os
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

# Declare vectorstore as a global variable
vectorstore = None

# Introduce a session state variable to track processing completion
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

def get_pdf_text(pdf_docs):
    data = []
    for pdf in pdf_docs:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf.read())
            tmp_file_path = tmp_file.name

        loader = PyMuPDFLoader(tmp_file_path)
        data.extend(loader.load())
        os.remove(tmp_file_path)

    return data

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(text)
    return all_splits

def get_vectorstore(text_chunks):
    global vectorstore
    print("Entered getvectorestore")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=text_chunks,
        collection_name="rag-chroma",
        embedding=embeddings,
    )
    print("Entered getvectorestore33")

def handle_userinput(user_question):
    global vectorstore
    print("Entered getvectorestore2345")
    if st.session_state.processing_complete:
        print("Entered getvectorestore76654")
        if vectorstore is not None:
            print("Entered getvectorestore33333")
            docs = vectorstore.similarity_search(user_question)
            finaldata = docs[2].page_content if len(docs) > 2 else ""
            finalresponse = getLLamaresponse(user_question, 200, finaldata)
            st.write(finalresponse)
        else:
            st.write("Vectorstore is not initialized")
    else:
        st.write("Processing is still ongoing.")

def getLLamaresponse(input_text, no_words, finaldata):
    llm = CTransformers(model='./data/llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens': 2000,
                                'temperature': 0.01})
    template = """
    Using these words {finaldata}
        Give clear and simple instructions in points for a common person on the topic '{input_text}'.
        The instructions should be within {no_words} words.
    """
    prompt = PromptTemplate(input_variables=["input_text", 'no_words', 'finaldata'], template=template)
    response = llm(prompt.format(input_text=input_text, no_words=no_words, finaldata=finaldata))
    return response

def main():
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                st.session_state.processing_complete = False  # Reset processing flag
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vectorstore(text_chunks)
                st.session_state.processing_complete = True  # Update processing flag after completion

if __name__ == '__main__':
    main()
