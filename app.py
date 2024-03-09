#important libraries wrt langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
#embedding technique / converting pdf to vectors
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import generativeai as ai
#vector embeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
#for chatting and defining prompts
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
#importing our env 
import streamlit as st
from PyPDF2 import PdfReader #helps to read all the document
import os
from dotenv import load_dotenv
import subprocess


def install_requirements(requirements_file):
    try:
        # Construct the pip install command
        pip_install_cmd = ['pip', 'install', '-r', requirements_file]
        
        # Run the pip install command
        subprocess.run(pip_install_cmd, check=True)
        
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print("Error installing dependencies:", e)
        # You can handle the error as needed here


#Main Code
load_dotenv()
#configure API Key
ai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


#Read the PDf ,Go through all pages, Extract the text
def get_pdf_text(pdf_docs):
    text=''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text         

# Dividing the whole text into 1000 chunks/tokens , where overlap of 100 can be possible
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 5000 , chunk_overlap = 500)
    tokens = text_splitter.split_text(text)
    return tokens

#converting tokens/chunks to vectors
def get_vector_store(tokens):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key="AIzaSyB4QNli0UOBec0r0LX9pg1Vb7XnY_xDKNI")
    # Ensure the directory exists
    index_directory = 'faissindex'
    os.makedirs(index_directory,exist_ok=True)
    
    # Save the vector store
    vector_store = FAISS.from_texts(tokens, embedding=embeddings)
    vector_store.save_local(os.path.join(index_directory, 'index.faiss'))

def give_prompt():
    prompt_template = '''
    Believe you are true expert in whatever question asked, and answer the question as detailed as possible in the provided context only,
    make sure to check the whole document correctly before answering all the details,
    if the answer is not available in the context just say "Answer Not Available", Don't provide the wrong answer
    Context: \n{context}?\n
    Question: \n{question}\n
    
    Answer: 
    '''
    model = ChatGoogleGenerativeAI(model='gemini-pro',temperature=0.2,max_output_tokens=1000,verbose=True)
    prompt = PromptTemplate(template=prompt_template,input_variables=['Context','Question'])
    #chain_type = 'stuff' helps to text summarization
    chain = load_qa_chain(model,chain_type = 'stuff',prompt = prompt , verbose=True)
    return chain

def input(question):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = None
    
    # Check if the file exists before trying to load the Faiss index
    index_path = 'faissindex/index.faiss'
    if os.path.exists(index_path):
        new_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"Error: The file {index_path} does not exist or is not a valid Faiss index directory.")

    # Check if new_db is not None before using it
    if new_db is not None:
        doc = new_db.similarity_search(question)
        chain = give_prompt()

        response = chain(
            {
                "input_documents": doc, "question": question
            },
            return_only_outputs=True)

        print(response)
        st.write("Reply: ", response["output_text"])
    else:
        st.error('Please build the database first.')

    

def page_configure():
    st.set_page_config(
        page_title="ChatPDF AI",
        page_icon="📝",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    st.title("Chat With Your PDF")
     


def main():
    page_configure()
    # Inject JavaScript using st.markdown
    st.markdown("""
        <script>
            // Your JavaScript code here
            axios.get('your_api_endpoint', { timeout: 15000 })
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
    
    # Sidebar
    st.sidebar.title("ChatPDF AI")
    st.sidebar.write("Upload PDF Documents:")
    uploaded_files = st.sidebar.file_uploader(" ", type=["pdf"], accept_multiple_files=True)
    question = st.sidebar.text_input("Enter your question:")

    if st.sidebar.button("Get Answer"):
        if not uploaded_files:
            st.sidebar.warning("Please upload at least one PDF document.")
        elif not question:
            st.sidebar.warning("Please enter a question.")
        else:
            with st.spinner("Fetching Answer..."):
                # Read PDFs and extract text
                pdf_texts = [get_pdf_text([pdf]) for pdf in uploaded_files]
                # Combine text from multiple PDFs
                combined_text = ' '.join(pdf_texts)
                # Split text into chunks
                text_chunks = get_text_chunks(combined_text)

                # Create and save vector store
                get_vector_store(text_chunks)

                # Get the answer using the LangChain pipeline
                input(question)
                st.success("Answer retrieved successfully!")

if __name__ == "__main__":
    main()
    requirements_file = "requirements.txt"
    # Install requirements
    install_requirements(requirements_file)


