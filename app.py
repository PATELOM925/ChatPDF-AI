#important libraries wrt langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import generativeai as ai 
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI  
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate     
from langchain_core.language_models import BaseLanguageModel
import streamlit as st.  
from PyPDF2 import PdfReader 
import os  
import subprocess
from dotenv import load_dotenv

def install_requirements(requirements_file):
    try: # Construct the pip install command
        pip_install_cmd = ['pip', 'install', '-r', requirements_file] # Run the pip install command
        subprocess.run(pip_install_cmd, check=True)     
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print("Error installing dependencies:", e) # You can handle the error as needed here

# Load environment variables
load_dotenv()
ai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Read the PDF and extract text
def get_pdf_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text         

# Split text into chunks for processing
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    tokens = text_splitter.split_text(text)
    return tokens

# Create embeddings from text chunks
def create_embeddings(tokens):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=os.getenv('GOOGLE_API_KEY'))
    vector_store = FAISS.from_texts(tokens, embedding=embeddings)
    return vector_store

# Define the prompt template and load the QA chain
def give_prompt():
    prompt_template = '''
    Believe you are true expert in whatever question asked, and answer the question as detailed as possible in the provided context only,
    make sure to check the whole document correctly before answering all the details,
    if the answer is not available in the context just say "Answer Not Available", Don't provide the wrong answer
    Context: \n{context}?\n
    Question: \n{question}\n
    Answer: 
    '''
    
    model = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.2, max_tokens=1000)
    
    # Ensure that ChatGoogleGenerativeAI is compatible with LangChain's requirements.
    if not isinstance(model, BaseLanguageModel):
        st.error("The model is not compatible with LangChain.")
        return None
    
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt, verbose=True)
    return chain

def page_configure():
    st.set_page_config(
        page_title="ChatPDF AI",
        page_icon="📝",
        layout="centered",
        initial_sidebar_state="expanded",)
    st.title("Chat With Your PDF")

def main():
    page_configure()
    st.markdown("""
            <script>
                // Your JavaScript code here
                axios.get('your_api_endpoint', { timeout: 20000 })
                    .then(response => {
                        // handle the response
                        console.log(response);
                    })
                    .catch(error => {
                        // handle the error
                        console.error(error);
                    });
            </script>
        """, unsafe_allow_html=True)  
    
    st.sidebar.title("ChatPDF AI")
    uploaded_files = st.sidebar.file_uploader("Upload PDF(s):", type=["pdf"], accept_multiple_files=True)
    question = st.sidebar.text_input("Enter your question:")

    if st.sidebar.button("Get Answer"):
        if not uploaded_files:
            st.sidebar.warning("Please upload at least one PDF document.")
        elif not question:
            st.sidebar.warning("Please enter a question.")
        else:
            with st.spinner("Fetching Answer..."):
                pdf_texts = [get_pdf_text([pdf]) for pdf in uploaded_files]  # Read PDFs and extract text
                combined_text = ' '.join(pdf_texts)  # Combine text from multiple PDFs
                text_chunks = get_text_chunks(combined_text)  # Split text into chunks
                
                # Create embeddings directly without saving/loading from disk
                vector_store = create_embeddings(text_chunks)  
                
                # Allow users to ask a question immediately after processing
                if question:
                    with st.spinner("Searching..."):
                        try:
                            docs = vector_store.similarity_search(question)
                            if docs:
                                chain = give_prompt()  # Get the chain
                                if chain is None:
                                    st.error("Failed to initialize the prompt chain.")
                                    return  # Exit early if chain is not valid

                                response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
                                st.write("Reply:", response.get("output_text", "Error generating response."))
                            else:
                                st.warning("No relevant information found.")
                        except Exception as e:
                            st.error(f"Error processing your query: {e}")

if __name__ == "__main__":
    main()
    requirements_file = "requirements.txt"
    install_requirements(requirements_file) # Install requirements
