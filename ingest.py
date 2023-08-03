from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import os
from constants import CHROMA_SETTINGS

persist_directory = 'db'
embedding_model = 'all-MiniLM-L6-v2'

def main():
    # Extract documents
    for root, dirs, files in os.walk('docs'):
        for file in files:
            if file.endswith('.pdf'):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))
    # Load and split
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    # Create embeddings
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
    # Create vector store here
    db = Chroma.from_documents(texts, 
                               embeddings, 
                               persist_directory=persist_directory,
                               client_settings=CHROMA_SETTINGS)
    db.persist()
    db=None

if __name__ == "__main__":
    main()