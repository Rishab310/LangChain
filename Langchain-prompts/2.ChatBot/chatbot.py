from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# llm = HuggingFacePipeline.from_model_id(
#     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
#     model_kwargs={"cache_dir": os.getenv("HUGGINGFACE_CACHE_PATH")},
#     pipeline_kwargs={
#         "temperature": 0.1,
#         # "max_new_tokens": 500,
#         "return_full_text": False,
#     }
# )

# model = ChatHuggingFace(llm=llm)

model = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
    model="openai/gpt-oss-20b",   # or "openai/gpt-oss-120b"
    temperature=0,
    max_retries=3,
    reasoning_effort="low",   # "low" | "medium" | "high"
    max_tokens=100,    
)

chat_history = [
    SystemMessage(content='You are an AI assistant, answer all queries in one line only.')
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit' :
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ",result.content)

print(chat_history)