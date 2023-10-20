import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

from PyPDF2 import PdfReader

from langchain.llms import LlamaCpp, VertexAI
# Todo change to Vertex AI
from langchain.chat_models import ChatVertexAI,ChatOpenAI
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.text_splitter import  CharacterTextSplitter
from langchain.chains import  ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.embeddings import VertexAIEmbeddings
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings

import hashlib
from tqdm import tqdm
from functools import lru_cache

from pymongo import MongoClient
import certifi

import os

os.environ["SENTENCE_TRANSFORMERS_HOME"] = "tmp/st/"

client = MongoClient("mongodb+srv://ashwin:pswd123ash@retail-demo.2wqno.mongodb.net/?retryWrites=true&w=majority",
                     tlsCAFile=certifi.where())
db = client["sample"]

one_way_hash = lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()

CHAT_VERIFY_COL = "chatverify_new"
CHAT_APP_COL = "chatapp_new"

def check_doc_in_mdb(md5):
    if len(list(db[CHAT_VERIFY_COL].find({"md5": md5}))) > 0:
        return True
    else:
        return False
    
def insert_doc_verify_mdb(md5):
    db[CHAT_VERIFY_COL].insert_one({"md5": md5})

def get_pdf_data(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    md5 = one_way_hash(text)
    for page in pdf_reader.pages:
        text += page.extract_text()
    if check_doc_in_mdb(md5):
        return None, None
    else:
        return text,md5


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_embeddings_transformer():
    embeddings = VertexAIEmbeddings()
    return embeddings

# def get_embeddings_transformer():
#     return HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")

@lru_cache(maxsize=1)
def get_vector_store():
    col = db[CHAT_APP_COL]
    vs = MongoDBAtlasVectorSearch(collection=col, embedding=get_embeddings_transformer(), index_name="default",
                                  embedding_key="vec", text_key="line")
    return vs

@lru_cache(maxsize=1)
def get_conversation_chain():
    llm = ChatVertexAI()
    # llm = ChatOpenAI()
    retriever = get_vector_store().as_retriever(search_type="mmr", search_kwargs={'k': 10, 'lambda_mult': 0.25})
    memory = ConversationBufferWindowMemory(memory_key='chat_history', k=5, return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    conv = get_conversation_chain()
    response = conv({'question': user_question, "chat_history": st.session_state.chat_history})
    chat_history = [{"user": response["question"], "assistant": response["answer"]}]
    st.session_state.chat_history += chat_history
    for i, message in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(message["user"])
        with st.chat_message("assistant"):
            st.markdown(message["assistant"])
        # if i % 2 == 0:
        #     with st.chat_message("user"):
        #         st.markdown(message)
        # else:
        #     with st.chat_message("assistant"):
        #         st.markdown(message.content)



def main():
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    
    st.markdown(
        """<img src="https://lh3.googleusercontent.com/I2_PSO0vMM8kLJxJ-OUIqtSBo3krzhmctqIkFv8Exgchm5X04h_MysTSB-8mELD6J_OIA1N2ExP_=e14-rj-sc0xffffff-h338-w600" class=" css-1lo3ubz" alt="MongoDB logo" style="height:200px;width:340px;align:center"> """,
        unsafe_allow_html=True)
    # st.title("""Assistant for any source powered by Atlas Vector Search and VertexAI""")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Assistant for any source powered by MongoDB Atlas Vector Search and VertexAI")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        print(">>>>>>>>>>>>>>>")
        handle_userinput(user_question)

    # st.session_state.conversation = get_conversation_chain(get_vector_store())

    with st.sidebar:
        st.title("Process your PDFs and perform vector search")
        st.markdown('''
        ## About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        - [MongoDB Vector Search](https://www.mongodb.com/products/platform/atlas-vector-search)
        ''')
        add_vertical_space(3)
        st.subheader("Your documents")
        pdf = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=False)
        if st.button("Process"):
            vs = get_vector_store()
            with st.spinner("Processing"):
                # get pdf text
                raw_text, md5 = get_pdf_data(pdf)

                if raw_text:
                    # get the text chunks
                    text_chunks = get_text_chunks(raw_text)
                    if len(text_chunks)>1000:
                        split = 100
                    else:
                        split = 10
                    for i in range(0, len(text_chunks), split):
                        batch_chunks = text_chunks[i:(i + split-1)]
                        vs.add_texts(batch_chunks)
                # insert to md5 once indexed
                insert_doc_verify_mdb(md5)
            
            st.write('Document added successfully')
        st.write('Made with ❤️ by [Ashwin Gangadhar](linkedin.com/in/ashwin-gangadhar-00b17046) and [Venkatesh Shanbhag](https://www.linkedin.com/in/venkatesh-shanbhag/)')

    add_vertical_space(5)

if __name__ == "__main__":
    main()
