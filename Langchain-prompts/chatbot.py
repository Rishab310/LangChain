from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    model_kwargs={"cache_dir": "../hf_cache"},
    pipeline_kwargs={
        "temperature": 0.1,
        # "max_new_tokens": 500,
        "return_full_text": False,
    }
)

model = ChatHuggingFace(llm=llm)

while True:
    user_input = input('You: ')
    if user_input == 'exit' :
        break
    result = model.invoke(user_input)
    print("AI: ",result.content)