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


# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
) 

# 2nd prompt -> summary of the report

template2 = PromptTemplate(
    template='Write a five line summary on the following text. /n {text}',
    input_variables=['text']
) 

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser 

result = chain.invoke({
    'topic' : 'Black hole'
})

print(result)