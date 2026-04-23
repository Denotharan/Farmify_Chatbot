import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

import time


load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(path):
    text = ""

    pdf_reader = PdfReader(path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=700)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):

    if os.path.exists("faiss_index"):
        st.info("Loading existing index from disk...")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = None


    batch_size = 15
    sleep_time = 20

    total_chunks = len(text_chunks)
    st.write(
        f"Processing {total_chunks} chunks. This will take about {int((total_chunks / batch_size) * sleep_time)} seconds...")

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(0, total_chunks, batch_size):

        batch = text_chunks[i: i + batch_size]

        try:
            if vector_store is None:
                vector_store = FAISS.from_texts(batch, embedding=embeddings)
            else:
                vector_store.add_texts(batch)

            #visual progress
            current_progress = min((i + batch_size) / total_chunks, 1.0)
            progress_bar.progress(current_progress)
            status_text.text(f"Processed batch {i // batch_size + 1}/{(total_chunks // batch_size) + 1}")


            time.sleep(sleep_time)

        except Exception as e:

            if "429" in str(e):
                st.warning("Rate limit hit. Cooling down for 60 seconds...")
                time.sleep(60)

                if vector_store is None:
                    vector_store = FAISS.from_texts(batch, embedding=embeddings)
                else:
                    vector_store.add_texts(batch)
            else:
                st.error(f"Error: {e}")
                break

    if vector_store:
        vector_store.save_local("faiss_index")
        st.success("Knowledge Base successfully created and saved!")
def get_retrieval_chain(memory):
    prompt_template = PromptTemplate(
        template="""
        Check for the answer in the provided chat history. If not available then, check the context.
        Answer the question based on the provided context. Break down the answer into steps in detail unless asked otherwise.
        If the answer is not in the context, say: "answer is not available in the context".
        
        Previous Conversation:
        {chat_history}
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """,
        input_variables=["chat_history", "context", "question"]
    )
    
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    retriever = FAISS.load_local("faiss_index", GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"), allow_dangerous_deserialization=True).as_retriever()
    chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever, chain_type="stuff", chain_type_kwargs={"prompt": prompt_template, "memory": memory})
    return chain

def user_input(user_question):

    chain = get_retrieval_chain(st.session_state.memory)
    response = chain.invoke(user_question)['result']
    return response


@st.cache_resource
def process_data_and_create_index(file_path):
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return False

    # 1. Get Text
    raw_text = get_pdf_text(file_path)

    # 2. Get Chunks

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    text_chunks = text_splitter.split_text(raw_text)

    # 3. Create Vector Store
    try:
        get_vector_store(text_chunks)
        return True
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        return False

def main():
    
    st.set_page_config("FARMIFY", layout = 'wide', initial_sidebar_state="expanded")
    st.title("FARMIFY")
    st.image("3971149copy.jpg")
    st.logo("FARMIFY.png",size="large")
    st.sidebar.write("BY:  \nDENO                   \nMANO ")
    pdf_path = "HORTICULTURE.pdf"  #-------------------------------------------------------------PATH-----


    user_question = st.chat_input("Ask a Question to me!!!")
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")
        st.session_state['chat_history'] = []
    
    if user_question:
        instance = {'user': user_question}
        response = user_input(user_question)
        instance['bot'] = response
        st.session_state.chat_history.append(instance)
    
    for index, chat in enumerate(st.session_state.chat_history):
        is_last = index == len(st.session_state.chat_history) - 1 
        if "user" in chat:
            with st.chat_message("user"):
                st.write(chat["user"], unsafe_allow_html=True)
        
        if "bot" in chat:
            with st.chat_message("assistant"):
                if is_last:
                    def stream_data():
                        for word in str(chat["bot"]).split(" "):
                            yield word + " "
                            time.sleep(0.02)
                    st.write_stream(stream_data)
                else:
                    st.write(chat["bot"])

        
if __name__ == "__main__":
    main()

