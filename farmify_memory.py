import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from gtts import gTTS
import time

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def get_pdf_text(path):
    text = ""
    pdf_reader = PdfReader(path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_retrieval_chain(memory):
    prompt_template = PromptTemplate(
        template="""
        Answer the question based on the provided context. 
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
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
    retriever = FAISS.load_local("faiss_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True).as_retriever()
    chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever, chain_type="stuff", chain_type_kwargs={"prompt": prompt_template, "memory": memory})
    return chain

def user_input(user_question):
    chain = get_retrieval_chain(st.session_state.memory)
    response = chain.run(user_question)
    return response

def main():
    st.set_page_config("FARMIFY", layout='wide', initial_sidebar_state="expanded")
    st.title("FARMIFY")
    st.image("1696604795690.jpeg")
    st.sidebar.write("Write your credentials here")
    
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
