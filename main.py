from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import os
from transformers import GPT2Tokenizer
import streamlit as st

from dotenv import load_dotenv;load_dotenv() # openai_key  .env 선언 사용 

# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# openai.api_key= st.secrets["OPENAI_API_KEY"]
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def tiktoken_len(text):
    tokens = tokenizer.tokenize(text)
    return len(tokens)


loader = TextLoader("unsu.txt", encoding='utf-8')
documents = loader.load()
# documents[0].page_content=documents[0].page_content.replace("\n"," ")

text_splitter = RecursiveCharacterTextSplitter(
        chunk_size =100,
        chunk_overlap  = 0,
        separators=[". "],
        length_function =tiktoken_len
    )

pages = text_splitter.split_documents(documents)
index = FAISS.from_documents(pages, OpenAIEmbeddings())
index.save_local("faiss-unsu-txt")

llm_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)  
chain = load_qa_chain(llm_model, chain_type="stuff")


# query = "아내가 먹고 싶어하는 음식은?"
# docs = index.similarity_search(query)
# res = chain.run(input_documents=docs, question=query)
# print( query , res)



with st.sidebar:

    "[제주과학고](http://jeju-s.jje.hs.kr/)"
    "[Streamlit 사용법](https://zzsza.github.io/mlops/2021/02/07/python-streamlit-dashboard/)"
    "[docs.streamlit.io](https://docs.streamlit.io/)"


st.header(f":LightGray[운수좋은 날]", divider='gray')

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "hello !"}]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

    # if not openai.api_key:
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()

    
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    query = prompt
    docs = index.similarity_search(query)
   

    
    with st.spinner("Waiting ..."):
       respons = chain.run(input_documents=docs, question=query)
    
    st.session_state.messages.append({"role": "assistant", "content":  respons} )
    st.chat_message("assistant").write(respons)
