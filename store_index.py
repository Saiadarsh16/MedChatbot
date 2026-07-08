print("***** STORE_INDEX STARTED *****")
from dotenv import load_dotenv
import os
from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings,
)
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

print("1. Loading environment...")
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

print("2. Loading PDFs...")
extracted_data = load_pdf_file("data/")

print("Loaded", len(extracted_data), "pages")

print("3. Filtering...")
filter_data = filter_to_minimal_docs(extracted_data)

print("Filtered:", len(filter_data))

print("4. Splitting...")
text_chunks = text_split(filter_data)

print("Chunks:", len(text_chunks))

print("5. Loading embedding model...")
embeddings = download_hugging_face_embeddings()

print("Embedding model loaded.")

print("6. Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-bot"

if not pc.has_index(index_name):
    print("Creating index...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
    )

print("Opening index...")
index = pc.Index(index_name)

print("7. Uploading documents...")

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name,
)

print("Done!")