from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os 

load_dotenv()

model = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
    model="openai/gpt-oss-20b",
    temperature=0,
    max_retries=3,
    reasoning_effort="low",  # "low" | "medium" | "high"
    max_tokens=200,
)

prompt = PromptTemplate(
    template='Write a summary for the following poem -\n{poem}',
    input_variables=['poem']
)

parser = StrOutputParser()

loader = TextLoader('Langchain-Document_Loaders/Files/cricket.txt', encoding='utf-8')

docs = loader.load()

print(docs[0].metadata)

chain = prompt | model | parser

result = chain.invoke({
    'poem': docs[0].page_content
})

print(result)