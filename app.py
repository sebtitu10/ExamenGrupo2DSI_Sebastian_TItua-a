import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory #memoria del chat
from langchain.chains import ConversationalRetrievalChain #perimite charlar con vectorstore
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

def get_pdf_text(pdf_docs):
    text = "" #alamacenamos en una cadena los PDFs.
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    #separador de texto
    text_splitter = CharacterTextSplitter(
        separator="\n",
        #tamanio del fragmento
        chunk_size=1000,
        #superposicion de fragmentos
        chunk_overlap=200,
        length_function=len
    )
    #creamos fragmentos a partir de nuestro divisor de texto
    chuncks = text_splitter.split_text(text) #dividimos y pasamos el texto
    return chuncks

def get_vectorstore(text_chuncks):
    embeddings = OpenAIEmbeddings()
    # creamos la base de datos,toma fragmentos y los embedings
    #vectorstore = FAISS.from_texts(texts=text_chuncks, embeddings=embeddings)
    vectorstore = FAISS.from_texts(texts=text_chuncks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key ='chat_history', return_messages= True)
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
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)



def main():
    load_dotenv()
    st.set_page_config(page_title="EXAMEN SEBAS ", page_icon=":books:") # ponemos el titulo a la pagina
    st.write(css, unsafe_allow_html=True) #activamos los estilos CSS
    #cuando uso seciones tengo que inicializar
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Pregunta lo que quieras")#encabezado
    user_question = st.text_input("Realiza una pregunta")# un simple label

    if user_question:
        handle_userinput(user_question)


    with st.sidebar:  #barra lateral
        st.subheader("Tus documentos")
        pdf_docs = st.file_uploader("carga tus informacion  y da click en procesar",accept_multiple_files=True) #cargador de archivos
        if st.button("Procesar"):
            with st.spinner("Procesando"):
              #get pdf texto
                raw_text = get_pdf_text(pdf_docs)
                st.write(raw_text)
              # get text chuncks
                text_chunks = get_text_chunks(raw_text) #pasamos una cadena y devuelve una lista de fragmentos
                st.write(text_chunks)
              #crear vector store
                vectorstore = get_vectorstore(text_chunks)
              #cadena de conversacion
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__=='__main__':
    main()
