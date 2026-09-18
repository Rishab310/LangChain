from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=os.getenv("HUGGINGFACE_CACHE_PATH")
)

sentences = ['This is an example sentence', 'Each sentence is converted']

vector = embedding.embed_documents(sentences)

print(str(vector))