from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
import os 

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Explain the following joke in short \n {joke}',
    input_variables=['joke']
)

model = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
    model="openai/gpt-oss-20b",
    temperature=0.5,
    max_retries=3,
    reasoning_effort="low",  # "low" | "medium" | "high"
    max_tokens=100,
)

parser = StrOutputParser()
joke_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'explanation': RunnableSequence( prompt2, model, parser)
})

final_chain = RunnableSequence(joke_chain, parallel_chain)

result = final_chain.invoke({
    'topic' : 'AI'
})

print(result)
print(result['joke'])
print(result['explanation'])
