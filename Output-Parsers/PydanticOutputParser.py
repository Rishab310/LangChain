from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os 

load_dotenv()

model = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
    model="openai/gpt-oss-20b",
    temperature=0.5,
    max_retries=3,
    reasoning_effort="low",  # "low" | "medium" | "high"
    max_tokens=100,
)

class Person(BaseModel):
    name : str = Field(description="The person's name")
    age : int = Field(description="The person's age")
    city : str = Field(description="The city where the person lives")
    profession : str = Field(description="The field in with this person works")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Give me the name, age, city and person of this famous person : {person} \n {format_instruction}',
    input_variables=['person'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({
    'person' :'Anushka Sharma'
})

print(result)