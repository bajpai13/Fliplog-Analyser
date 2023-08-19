# Load environment variables from a .env file
from dotenv import load_dotenv

# Import the Streamlit library for building web applications
import streamlit as st

# Import the pandas library for working with data in tables
import pandas as pd

# Import PdfReader from PyPDF2 to read PDF files
from PyPDF2 import PdfReader

# Import the CharacterTextSplitter class from langchain for splitting text
from langchain.text_splitter import CharacterTextSplitter

# Import the HuggingFaceEmbeddings class from langchain for embeddings
from langchain.embeddings import HuggingFaceEmbeddings

# Import the FAISS vector store class from langchain
from langchain.vectorstores import FAISS

# Import the PromptTemplate class from langchain for prompts
from langchain.prompts import PromptTemplate

# Import AutoGPTQForCausalLM for automatic query generation
from auto_gptq import AutoGPTQForCausalLM

# Import hf_hub_download for downloading models from Hugging Face Hub
from huggingface_hub import hf_hub_download

# Import HuggingFacePipeline and LlamaCpp classes from langchain
from langchain.llms import HuggingFacePipeline, LlamaCpp

# Import necessary classes from the transformers library
from transformers import (
    AutoModel, 
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline
)

# Import constants from an external file
from constants import EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME

# Specify the path to the downloaded MiniLM-L6-v2 model
model_path_embedding = "path_to_downloaded_model_directory"  # Replace with the actual path

# Initialize the HuggingFace model and tokenizer from the downloaded directory
model_embedding = AutoModel.from_pretrained(model_path_embedding)
tokenizer = AutoTokenizer.from_pretrained(model_path_embedding)

# Initialize the HuggingFaceEmbeddings with the model and tokenizer
embedding_model = HuggingFaceEmbeddings(model=model_embedding, tokenizer=tokenizer)

# Specify the name of the Alpaca LLM model
llm_model_name = "alpaca-7b-native-enhanced"

# Specify the directory where you have downloaded and extracted the Alpaca model
model_dir = "write_the_location_of_the_downloaded_model_here"

# Define the log search query with different log levels
logSearch = "Error,Warning,Critical,Alert,Emergency,Debug"

# Define the prompt for the LLN model to analyze log data and ruleset
llmPrompt = "Analyze the provided log data and ruleset to assess the compliance of our system with security policies and standards. Create a table summarizing the key findings, including any security violations, access control issues, or user privilege concerns. Ensure the table includes relevant details as Timestamp,Log Level,Log Entry,Rule Violation,User Information,Actionable Insights"

# Function to load the local Alpaca LLM model
def load_model(model_dir):
    # Specify the Alpaca model ID
    model_id = llm_model_name
    
    # Load the Alpaca LLM tokenizer from the provided model directory
    tokenizer = LlamaTokenizer.from_pretrained(model_dir)
    
    # Load the Alpaca LLM model from the provided model directory
    model = LlamaForCausalLM.from_pretrained(model_dir)
    
    # Load generation configuration for text generation
    generation_config = GenerationConfig.from_pretrained(model_dir)
    
    # Create a pipeline for text generation using the loaded model and tokenizer
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )
    
    # Create a local LLN pipeline using the HuggingFacePipeline class
    local_llm = HuggingFacePipeline(pipeline=pipe)
    
    return local_llm


# Text Spilter
def getTextChunks(text):
    text_splitter = CharacterTextSplitter(separator=["\n", " ", "\n\n", ""],chunk_size = 1000,chunk_overlap  = 200,length_function = len )
    chunks = text_splitter.split_text(text)
    st.write(chunks)
    return chunks

# Text/plain (fileToChunks)
def textToText(doc):
    loader = doc.read()
    text = str(loader)
    return text

# Application/pdf (fileToChunks)
def pdfToText(doc):
    text = ""
    loader = PdfReader(doc)
    for page in loader.pages:
        text += page.extract_text()
    return text

# Pdf/Text/Csv (fileToChunks)
def csvToText(doc):
    loader = doc.read()
    text = str(loader)
    return text

def fileToChunks(docs):
    text = ""
    for i in docs:
        if(i.type == "text/plain"):
            text += textToText(i)
        elif (i.type == "application/pdf"):
            text += pdfToText(i)
        elif (i.type == "text/csv"):
            text += csvToText(i)
    chunks = getTextChunks(text)
    return chunks

# Convert list into embedding (all-MiniLM-L6-v2 model) for log files
def getVectorStore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorStore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorStore

# Convert list into embedding (all-MiniLM-L6-v2 model) for rule sets
def get_vector_storeOpenAI(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorStore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorStore

# To Load the Model on the local machine
llmodel = load_model(model_dir)

# To get the output from the local llm model
def llmOutput(result, llmPrompt):
    QA_input = {
        'question': llmPrompt,
        'context': result
    }
    tempRes = llmodel([QA_input])
    res = list(tempRes["answer"])
    return res

def main():
    # Load environment variables from a .env file
    load_dotenv()

    # Configure the page settings for the Streamlit app
    st.set_page_config(page_title="FlipLog Analyser", page_icon=":open_file_folder:")

    # Display a header in the app
    st.header("Compliance Report")
    
    # Initialize empty lists to store chunks of rulebook,log files and compliance report data
    ruleBookChunks = []
    logChunks = []
    complianceReport = []
    
    # Create a sidebar in the app
    with st.sidebar:

        # Display a header in the sidebar
        st.header("*Upload Required Documents*")

        # Display a subheader and provide a file uploader for rulebook files
        st.subheader("Rulebook File")
        ruleBookFile = st.file_uploader("File types : Pdf, Csv, Txt", key="ruleBookFile", type=["pdf", "csv", "txt"], accept_multiple_files=True)
        
        # Display a subheader and provide a file uploader for log files
        st.subheader("Log File")
        logFile = st.file_uploader("File types : Pdf, Csv, Txt", key="logFile", type=["pdf", "csv", "txt"], accept_multiple_files=True)
        
        # Display a button labeled "Process"
        if st.button("Process"):
            # Display a spinner with the text "Processing"
            with st.spinner("Processing"):

                # Divide the rulebook file into chunks for processing
                ruleBookChunks = fileToChunks(ruleBookFile)
                # Divide the log file into chunks for processing
                logChunks = fileToChunks(logFile)

                # Obtain vector representations for the chunks from the rulebook 
                vectorStoreRule = get_vector_storeOpenAI(ruleBookChunks)
                # Obtain vector representations for the chunks from the log files
                vectorStorelog = getVectorStore(logChunks)

                # Generate an embedding for the log search query using an embedding model
                embedding_prompt = embedding_model.embed_query(logSearch)

                # Perform a similarity search on the log vector store using the embedding of the search query
                logOutput = vectorStorelog.similarity_search_by_vector(embedding_prompt)

                # Initialize an empty list to store the final results
                result = []

                # Iterate through each log output vector and find similar vectors in the rule vector store
                for i in logOutput:
                    # Perform a similarity search in the rule vector store for each log vector
                    temp = (vectorStoreRule.similarity_search_by_vector(i))
                    # Append the log vector and its corresponding rule vectors to the result list
                    result.append(i)
                    result.append(temp)

                # Getting output from the local llm model  
                for i in result:
                    # Generate LLN output using the llmOutput function and the llmPrompt
                    opll = llmOutput(i, llmPrompt)
                    # Append the LLN output to the compliance report list
                    complianceReport.append(opll)

            # Display a success message indicating that processing is done
            st.success('Processing is Done!')
    
    # Check if the "View Compliance Report" button is clicked
    if (st.button("View Compliance Report")):
        # Display a spinner while generating the report
        with st.spinner('Generating Report...'):
    
        # Create a data set table using the complianceReport data
        data_set_table = complianceReport

        # Create a pandas DataFrame from the data set table
        df = pd.DataFrame(data_set_table, columns=["Timestamp", "Log Level", "Log Entry", "Rule Violation", "User Information", "Actionable Insights"])
        
        # Display a success message indicating that the report is ready
        st.success('Compliance Report is Ready!')
        # Display the DataFrame as a table in the Streamlit app
        st.table(df)
# Call the main function if this script is executed as the main program        
if __name__ == '__main__':
    main()
