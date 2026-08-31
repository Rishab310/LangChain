from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="../hf_cache"
)

sentences = ['This is an example sentence', 'Each sentence is converted']

vector = embedding.embed_documents(sentences)

print(str(vector))