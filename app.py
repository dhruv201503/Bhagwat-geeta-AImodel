import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import base64
import textwrap
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from constants import CHROMA_SETTINGS

model_name = "LaMini-T5-738M"

model_id=f"MBZUAI/LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model_8bit = AutoModelForSeq2SeqLM.from_pretrained(model_id)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype = torch.float32
)

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
            'text2text-generation',
            model= base_model,
            tokenizer= tokenizer,
            max_length = 256,
            do_sample= True,
            temperature =0.3,
            top_p= 0.95,
            model_kwargs={
                "device_map": "auto","load_in_8bit": True
            }
    )

    local_llm= HuggingFacePipeline(pipeline = pipe)
    return local_llm

def qa_llm():
    llm =llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function = embeddings, client_settings= CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type ="stuff",
        retriever= retriever,
        return_source_documents=True
    )

    def process_answer(instruction):
        response =''
        instruction = instructionqa = qa_llm()
        generated_text = qa(instruction)
        answer = generated_text['result']
        return answer, generated_text
    
    def main():
        st.title('SEARCH YOUR PDF')
        with st.expander("About the app"):
            st.markdown(
                """
                This Generative AI answers you question about life by the teachings of Bhagwat Geeta
                """
            )

        question = st.text_area("Enter your question")
        if st.button("Search"):
            st.info("Your question: "+question)
            st.info("Your Answer")
            answer, metadata = process_answer(question)
            st.write(answer)
            st.write(metadata)

    if __name__ == '__main__':
        main()