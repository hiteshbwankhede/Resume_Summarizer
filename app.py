import streamlit as st
import tempfile
import os
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.mapreduce import MapReduceDocumentsChain
from summarizer import summarize_document_to_bullets_mapreduce

load_dotenv()



#.sidebar.header("Resume Summarizer")
pdf_file = st.sidebar.file_uploader("Upload Resume (PDF)", type=["pdf"])
num_points = st.sidebar.number_input("Number of Summary Points", min_value=1, max_value=10, value=5)

#st.set_page_config(layout="wide")
#st.title("Reliance Jio - Recruiter Feedback Performance System")
#t.markdown("<h1 style='text-align: center;'> Reliance Jio - Recruiter Feedback Performance System</h1>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: center;">
        <img src="https://e7.pngegg.com/pngimages/33/16/png-clipart-jio-logo-jio-reliance-digital-business-logo-mobile-phones-business-blue-text.png" width="30">
        <h2> Reliance - Resume Summarizer</h2>
    </div>
    """, 
    unsafe_allow_html=True
)

if pdf_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_file.read())
        temp_pdf_path = temp_pdf.name
    
    with st.spinner("Extracting text from resume..."):
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()
        
    
    with st.spinner("Generating summary..."):
        summary = summarize_document_to_bullets_mapreduce(documents, max_bullets=num_points)

    st.success("Summary Generated:")
    #formatted_summary = "\n- ".join([sentence.strip() for sentence in summary.split(". ") if sentence.strip()])
    #formatted_summary = "- " + formatted_summary  # Add a bullet point at the star
    #st.write(response["text"])
    #formatted_summary = summary
    for points in summary:
        st.write(points)