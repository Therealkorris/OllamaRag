import streamlit as st
import os
import time
import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.llms import Ollama
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Ensure directories exist
if not os.path.exists('files'):
    os.mkdir('files')
if not os.path.exists('chroma'):
    os.mkdir('chroma')

# Initialize or ensure session state variables exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'template' not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:"""

if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question")

if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(
        base_url="http://localhost:11434",
        model="mistral:instruct",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

if 'vectorstores' not in st.session_state:
    st.session_state.vectorstores = {}

st.title("PDF Chatbot")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload your PDF", type='pdf')
    mode_selection = st.radio("Choose Input Mode:", ("Text", "Speech"))
    chroma_directories = [name for name in os.listdir('chroma') if os.path.isdir(os.path.join('chroma', name))]
    selected_doc_name = st.selectbox("Select a document to query:", chroma_directories)

if uploaded_file is not None:
    file_path = f"files/{uploaded_file.name}"
    if not os.path.isfile(file_path):
        # Save the uploaded PDF file
        bytes_data = uploaded_file.getvalue()
        with open(file_path, "wb") as f:
            f.write(bytes_data)

        # Process and split the PDF document
        loader = PyPDFLoader(file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200, length_function=len)
        all_splits = text_splitter.split_documents(data)

        # Create a new Chroma vector store for this specific document
        doc_vectorstore = Chroma(persist_directory=f'chroma/{uploaded_file.name.split(".")[0]}',
                                 embedding_function=OllamaEmbeddings(model="mistral:instruct"))
        doc_vectorstore.add_documents(documents=all_splits)
        doc_vectorstore.persist()

        # Store the reference to the newly created vector store
        st.session_state.vectorstores[uploaded_file.name] = doc_vectorstore

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["message"])

# Handle user input based on selected mode
user_input = None
if mode_selection == "Text":
    user_input = st.text_input("Type your question here:", key="text_input")
elif mode_selection == "Speech":
    recognizer = sr.Recognizer()
    if st.button("Speak"):
        with sr.Microphone() as mic:
            st.info("Listening...")
            try:
                audio_data = recognizer.listen(mic, timeout=5)
                user_input = recognizer.recognize_google(audio_data)
                st.success(f"Recognized text: {user_input}")
            except sr.UnknownValueError:
                st.error("Google Speech Recognition could not understand the audio.")
            except sr.RequestError as e:
                st.error(f"Could not request results from Google Speech Recognition service; {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")


if user_input:
    user_message = {"role": "user", "message": user_input}
    st.session_state.chat_history.append(user_message)
    


    if selected_doc_name:
        # Check if the selected vector store is already loaded, if not, load it
        if selected_doc_name not in st.session_state.vectorstores:
            # Load the Chroma collection for the selected document
            doc_vectorstore = Chroma(persist_directory=f'chroma/{selected_doc_name}',
                                     embedding_function=OllamaEmbeddings(model="mistral:instruct"))
            # Update st.session_state.vectorstores with the loaded Chroma collection
            st.session_state.vectorstores[selected_doc_name] = doc_vectorstore
        else:
            # If already loaded, retrieve it directly
            doc_vectorstore = st.session_state.vectorstores[selected_doc_name]

        # Initialize or reuse the QA chain with the retrieved or loaded vector store
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type='stuff',
            retriever=doc_vectorstore.as_retriever(),
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory,
            }
        )
        
        with st.spinner("Assistant is typing..."):
            response = st.session_state.qa_chain(user_input)
            
        chatbot_message = {"role": "assistant", "message": response['result']}
        st.session_state.chat_history.append(chatbot_message)
        
        # Display the chatbot's response
        with st.chat_message("assistant"):
            st.markdown(response['result'])
            
        # Text-to-Speech for the chatbot's response
        tts = gTTS(text=response['result'], lang='en', slow=False)
        tts_audio = BytesIO()
        tts.write_to_fp(tts_audio)
        tts_audio.seek(0)
        st.audio(tts_audio, format='audio/mp3')
