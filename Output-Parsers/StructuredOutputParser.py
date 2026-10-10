from langchain_openai import ChatOpenAI
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os 

load_dotenv()

model = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
    model="openai/gpt-oss-20b",
    temperature=0.5,
    max_retries=3,
    reasoning_effort="low",  # "low" | "medium" | "high"
    max_tokens=200,
)

response_schemas = [
    ResponseSchema(name="name", description="The person's name"),
    ResponseSchema(name="age", description="The person's age"),
    ResponseSchema(name="city", description="The city where the person lives"),
    ResponseSchema(name="profession", description="The field in with this person works")
]
parser = StructuredOutputParser.from_response_schemas(response_schemas)

template = PromptTemplate(
    template='Give me the name, age, city and profession of this famous person : {person} \n {format_instruction}',
    input_variables=['person'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({
    'person' :'Rohit Sharma'
})

print(result)