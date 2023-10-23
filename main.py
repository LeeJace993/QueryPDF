import streamlit as st
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

import os
os.environ['OPENAI_API_KEY']='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'
# SYSTEM ENVORONMENTS
PINECONE_API_KEY='28c5d070-2401-4507-8307-52a36190e9cb'
PINECONE_API_ENV='gcp-starter'


st.set_page_config(page_title="Ask Questions about your papers",page_icon=":robot:")
st.header("Ask anything about your papers!")
st.markdown("*This WEBSITE driven by [LangChain](www.langchain.com) will help you to consult The [OpenAI ROBOT](https://openai.com) about anything on your papers!*")
st.image(image="wall_paper.jpg")

def get_query():
    input_text=st.text_area(label="",placeholder="Ask Any Questions...",key="query question")
    return input_text

# upload file
pdf = st.file_uploader("Upload your PDF", type="pdf")

# extract the text
if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=0,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    
    # Create Embeddings
    embeddings=OpenAIEmbeddings()
    pinecone.init(api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV)
    index_name="chatpdf"
    docsearch=Pinecone.from_texts(chunks,embeddings,index_name=index_name)
        #Get Query
    query = st.text_input("Ask a question about your PDF:")
    llm=OpenAI(temperature=0.5)
    chain=load_qa_chain(llm,chain_type="stuff")
    docs=docsearch.similarity_search(query)
    result=chain({"input_documents":docs,"question":query})
    st.write(result['output_text'])
            #with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                #tmp_file.write(file.read())
                #st.write(tmp_file)

                #fp = Path(tmp_file.name)
                #fp.write_bytes(file.getvalue())
                #with open(tmp_file.name, "rb") as f:
                #    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                #pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" ' \
                #              f'width="800" height="1000" type="application/pdf">'
                #st.markdown(pdf_display, unsafe_allow_html=True)






