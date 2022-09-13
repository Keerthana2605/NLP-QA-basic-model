from re import T
from unittest import result
import streamlit as st
import logging
from tqdm import tqdm
from haystack.utils import launch_es,clean_wiki_text, convert_files_to_docs, fetch_archive_from_http, print_answers, print_documents, print_questions
from haystack.nodes import FARMReader, TransformersReader, BM25Retriever, TfidfRetriever, QuestionGenerator, DensePassageRetriever
from haystack.document_stores import ElasticsearchDocumentStore, InMemoryDocumentStore
from haystack.pipelines import ExtractiveQAPipeline, DocumentSearchPipeline, QuestionGenerationPipeline, QuestionAnswerGenerationPipeline, RetrieverQuestionGenerationPipeline
from pprint import pprint


st.title("Question-Answering model on computer science from Haystack")
inp = st.text_input("Enter your question")
bt = st.button("Retrieve document")

txt = st.text_area("Enter your text")
bt1 = st.button("Generate question")

inp1 = st.text_input("Enter your question to retieve the document")
bt2 = st.button("Get answer")

#st.write(inp)

def fn_qa():


    launch_es()
    document_store = InMemoryDocumentStore() #host="localhost", username="", password="", index="document"
    doc_dir = "CSEdata"
    docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
    document_store.write_documents(docs)

    retriever = TfidfRetriever(document_store=document_store)
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
    #pipe = ExtractiveQAPipeline(reader, retriever)
    #prediction = pipe.run(
    #query=inp, params={"Retriever": {"top_k": 6}, "Reader": {"top_k": 3}})
    #pprint(prediction['answers'])
    pipeline = DocumentSearchPipeline(retriever)
    query = inp
    result = pipeline.run(query, params={"Retriever": {"top_k": 3}})
    #st.write(prediction['answers'])
    st.write(result)

if bt == True:
    fn_qa()


def fn_qg():
    launch_es()
    text = txt
    docs=[{"content":text}]
    document_store = InMemoryDocumentStore()
    document_store.write_documents(docs)
    question_generator = QuestionGenerator()
    question_generator_pipeline = QuestionGenerationPipeline(question_generator)
    for idn, document in  enumerate(document_store):
        result = question_generator_pipeline.run(documents = [document])
        st.write(result)

if bt1 == True:
    fn_qg()

def fn_dr():
    launch_es()
    document_store = InMemoryDocumentStore()
    doc_dir = "CSEdata"
    docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
    document_store.write_documents(docs)

    retriever = TfidfRetriever(document_store=document_store)
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
    p= ExtractiveQAPipeline(reader, retriever)
    query = inp1
    res = p.run(query, params={"Retriever": {"top_k": 2}})
    st.write(res)

if bt2 == True:
    fn_dr()
