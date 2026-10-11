from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
import os 

load_dotenv()

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
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

def word_counter(text):
    return len(text.split())

joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_cahin = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'word_count' : RunnableLambda(word_counter),
    'word_count_alt' : RunnableLambda(lambda x : len(x.split()))
})

final_chain = RunnableSequence(joke_gen_chain, parallel_cahin)

result = final_chain.invoke({
    'topic' : 'AI'
})

foramtted_result = """{} \nWordCount-{}\nWordCountAlt-{}""".format(result['joke'], result['word_count'], result['word_count_alt'])

print(foramtted_result)