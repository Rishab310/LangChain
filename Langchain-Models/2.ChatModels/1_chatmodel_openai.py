from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Temperature is a parameter that controls the randomness of the model's output. 
# A higher temperature (e.g., 0.8) will result in more random outputs (creative),
# While a lower temperature (e.g., 0.2) will make the output more focused and deterministic.
# Value above 1.0 is not recommended as it can lead to nonsensical outputs.


# Maximum completion tokens is a parameter that controls the maximum number of tokens 
# (words or word pieces) that the model can generate in its response.

# Max tokens vs Max completion tokens: 
# Max tokens:  is the total number of tokens in the input and output combined, 
# Max completion tokens:  is the maximum number of tokens that the model can generate in its response.

model = ChatOpenAI(model="gpt-4", temperature=0.7, max_completion_tokens=100)


result = model.invoke("What is the capital of india")

print(result.content)