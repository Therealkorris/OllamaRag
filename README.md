Streamlit application that allows users to ask questions about the content of an uploaded PDF document using either text or speech input.

Features:
- Upload a PDF document
- Choose between text or speech input for your questions
- Get answers to your questions based on the content of the PDF
- Hear the chatbot's response through text-to-speech functionality

How to Use
- Upload your PDF: Click the "Upload your PDF" button and select the PDF document you want to chat about.
  Choose Input Mode: Select between "Text" or "Speech" depending on how you want to provide your questions.
  
Text Input:
  If you choose "Text", type your question in the text box provided.
  Speech Input: If you choose "Speech", click the "Speak" button and speak your question clearly. The application will use speech recognition to convert your speech to text.
  Optional: Select Document: If you have uploaded multiple PDFs and want to switch between them, use the dropdown menu in the sidebar to select the specific document you want to query.
  Ask your question: Once you have provided your question, the chatbot will process it and retrieve relevant information from the uploaded PDF.
  Get Your Answer: The chatbot will display its answer in a chat-like interface and you can also hear the answer through text-to-speech.
  
Technical Details
  This application is built using Streamlit and leverages Langchain, a library for building conversational AI applications. It utilizes a pre-trained large language model (LLM) called Ollama to understand and respond to user queries. The     retrieved responses are based on a combination of document retrieval and prompting the LLM with the retrieved passages and the conversation history.

Dependencies:
Streamlit
Langchain
SpeechRecognition (sr)
gtts
PyPDFLoader
