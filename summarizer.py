import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Union

def summarize_document_to_bullets_mapreduce(document: Union[Document, List[Document]], 
                                          openai_api_key: str = None,
                                          max_bullets: int = 10) -> List[str]:
    """
    Summarize a document or list of documents into bullet points using Map-Reduce approach.
    
    This function uses LangChain's map-reduce summarization which:
    1. Maps: Summarizes each chunk of the document separately
    2. Reduces: Combines all chunk summaries into a final coherent summary
    
    Args:
        document (Document or List[Document]): Document(s) loaded using PyPDF loader
        openai_api_key (str, optional): OpenAI API key. If None, uses OPENAI_API_KEY env variable
        max_bullets (int): Maximum number of bullet points (default: 10)
    
    Returns:
        List[str]: List of bullet point summaries
    
    Raises:
        ValueError: If API key is not provided
        Exception: If summarization fails
    """
    
    # Set up OpenAI API key
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    elif not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OpenAI API key must be provided either as parameter or OPENAI_API_KEY environment variable")
    
    # Initialize OpenAI LLM
    llm = OpenAI(
        temperature=0.2,
        max_tokens=1000
    )
    
    # Prepare documents
    if isinstance(document, Document):
        docs = [document]
    else:
        docs = document
    
    # Split documents into chunks if they're too long
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Smaller chunks for better map-reduce performance
        chunk_overlap=100,
        length_function=len
    )
    
    # Split all documents
    split_docs = []
    for doc in docs:
        if len(doc.page_content) > 2000:
            chunks = text_splitter.split_text(doc.page_content)
            for chunk in chunks:
                split_docs.append(Document(page_content=chunk, metadata=doc.metadata))
        else:
            split_docs.append(doc)
    
    # Create custom prompts for map-reduce
    map_prompt = PromptTemplate(
        template="""
Summarize the following document chunk into key bullet points. Focus on the most important information:

{text}

Key points from this section:
""",
        input_variables=["text"]
    )
    
    combine_prompt = PromptTemplate(
        template="""
You are summarizing a document. Below are summaries from different sections of the document.

Create a final summary with exactly {max_bullets} bullet points that capture the most important information across all sections. 

Each bullet point should be:
- Concise but informative
- Focused on key insights or important facts
- Non-redundant (avoid repeating similar information)

Section summaries:
{text}

Final summary ({max_bullets} bullet points maximum):
""",
        input_variables=["text", "max_bullets"]
    )
    
    try:
        # Load the map-reduce summarization chain
        chain = load_summarize_chain(
            llm=llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=False
        )
        
        # Run the chain with custom combine_prompt_kwargs
        result = chain.run(input_documents = split_docs, max_bullets=max_bullets)
        
        # Parse the result into bullet points
        bullet_points = []
        lines = result.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith('•') or line.startswith('-') or line.startswith('*') or line.startswith('1.') or line[0].isdigit()):
                # Remove bullet symbols and numbering
                clean_point = line.lstrip('•-*0123456789. ').strip()
                if clean_point and len(clean_point) > 10:  # Avoid very short points
                    bullet_points.append(clean_point)
            elif line and len(line) > 15 and not any(line.lower().startswith(prefix) for prefix in ['summary:', 'key points:', 'bullet points:', 'final summary:']):
                bullet_points.append(line)
        
        # Ensure we don't exceed max_bullets
        return bullet_points[:max_bullets]
        
    except Exception as e:
        raise Exception(f"Error generating map-reduce summary: {str(e)}")

#from langchain.document_loaders import PyPDFLoader

# Load document
#loader = PyPDFLoader(r"resume.pdf")
#documents = loader.load()

# Get map-reduce summary
#summary = summarize_document_to_bullets_mapreduce(documents, max_bullets=10)
#print(summary)