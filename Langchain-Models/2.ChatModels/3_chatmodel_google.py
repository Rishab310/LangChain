from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv 

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=1.5)

print("Sending request...")

result = model.invoke("Write a 5 line poem on cricket!!")
 
print(result.text)