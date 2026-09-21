from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
    model="openai/gpt-oss-20b",   # or "openai/gpt-oss-120b"
    temperature=0,
    max_retries=3,
    reasoning_effort="low",   # "low" | "medium" | "high"
    max_tokens=100,    
)

messages= [
    SystemMessage(content="You are a helpful, assistane"),
    HumanMessage(content='Tell me about langchain in one line.')
]

result = model.invoke(messages)

messages.append(AIMessage(content = result.content))

print(messages)