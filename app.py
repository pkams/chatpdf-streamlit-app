import streamlit as st 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch 
import base64 
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma 
from langchain.llms import HuggingFacePipeline
from langchain.chain import RetrievalQA


checkpoint = "MBZUAI/LaMini-Flan-T5-77M"

model = pipeline('text2text-generation', model = checkpoint)