import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

from PyPDF2 import PdfReader

from langchain.llms import LlamaCpp, OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain, RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from functools import lru_cache
import hashlib
from tqdm import tqdm

from pymongo import MongoClient

import os

os.environ[
    "MONGO_CONNECTION_STR"] = "mongodb+srv://ashwin:pswd123ash@retail-demo.2wqno.mongodb.net/?retryWrites=true&w=majority"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "tmp/st/"

client = MongoClient(os.getenv("MONGO_CONNECTION_STR"))
db = client["sample"]

one_way_hash = lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()


def check_doc_in_mdb(md5):
    if len(list(db["chatverify"].find({"md5": md5}))) > 0:
        return True
    else:
        db["chatverify"].insert_one({"md5": md5})
        return False


def get_pdf_data(pdf_docs):
    for pdf in pdf_docs:
        text = ""
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        if check_doc_in_mdb(one_way_hash(text)):
            return None
        else:
            return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_emb_trnsformers():
    emb_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings_w = HuggingFaceEmbeddings(
        model_name=emb_model,
        cache_folder=os.getenv('SENTENCE_TRANSFORMERS_HOME')
    )
    return embeddings_w


def get_embeddings_transformer():
    return HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base", model_kwargs={"device": "cpu"})


def get_vector_store():
    col = db["chatapp"]
    vs = MongoDBAtlasVectorSearch(collection=col, embedding=get_embeddings_transformer(), index_name="default",
                                  embedding_key="vec", text_key="line")
    return vs


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.chat_message("user"):
                st.markdown(message.content)
        else:
            with st.chat_message("assistant"):
                st.markdown(message.content)


def get_llm_chain():
    # # Locally Hosted LLM
    # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    # llm_llama = LlamaCpp(
    #     model_path="models/ggml-model-q4_0.bin", callback_manager=callback_manager, verbose=True,
    #     n_ctx=1024
    # )
    llm = OpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.2)
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=get_vector_store().as_retriever())
    return chain


def main():
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")

    st.markdown(
        """<img src="https://webimages.mongodb.com/_com_assets/cms/kuyjf3vea2hg34taa-horizontal_default_slate_blue.svg?auto=format%252Ccompress" class=" css-1lo3ubz" alt="MongoDB logo" style="height:300px;width:500px;align:center"> """,
        unsafe_allow_html=True)
    st.title("""Your Q&A Assistant for any source powered by Atlas Vector Search """)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.title("🤗💬 Talk to your Data Powered by LLM's and Vector Search")
        st.markdown('''
        ## About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [MongoDB Vector Search](https://www.mongodb.com/products/platform/atlas-vector-search)

        ''')
        add_vertical_space(3)
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_data(pdf_docs)

                if raw_text:
                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)

                    for i in tqdm(range(len(text_chunks), 100)):
                        batch_chunks = text_chunks[i:i + 99]
                        # create vector store
                        get_vector_store().add_texts(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(get_vector_store())
        add_vertical_space(5)
        st.write('Made with ❤️ by [Ashwin Gangadhar](linkedin.com/in/ashwin-gangadhar-00b17046)')


if __name__ == "__main__":
    main()