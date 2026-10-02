from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as st 
from langchain_core.prompts import PromptTemplate, load_prompt
import os


load_dotenv()

# @st.cache_resource
# def load_model():
#     llm = HuggingFacePipeline.from_model_id(
#         model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#         task="text-generation",
#         model_kwargs={"cache_dir": os.getenv("HUGGINGFACE_CACHE_PATH")},
#         pipeline_kwargs={
#             "temperature": 0.1,
#             # "max_new_tokens": 500,
#             "return_full_text": False,
#         }
#     )
#     return ChatHuggingFace(llm=llm)

# model = load_model()

model = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
    model="openai/gpt-oss-20b",   # or "openai/gpt-oss-120b"
    temperature=0,
    max_retries=3,
    reasoning_effort="low",   # "low" | "medium" | "high"
    # max_tokens=100,    
)

st.header('Research Tool')

paper_input = st.selectbox( "Select Research Paper Name :", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style :", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length :", ["Short (1 paragraph)", "Medium (3 paragraphs)", "Long (5 paragraphs)"] )


template = load_prompt('/Users/rishabgoyal/Learning/Langchain/Langchain-prompts/1.PromptTemplate/template.json')

if st.button("Summarize"):
    chain = template | model
    result = chain.invoke({
        'paper_input' : paper_input,
        'style_input' : style_input,
        'length_input' : length_input
    })
    st.write(result.content)



# Directly using PromptTemplate

# template = PromptTemplate(
#     template="""
# Please summarize the research paper titled "{paper_input}" with the following specifications:
# Explanation Style: {style_input}
# Explanation Length: {length_input}
# 1. Mathematical Details: 
#     - Include relevant mathematical equations if present in the paper. 
#     - Explain the mathematical concepts using simple, intuitive code snippets where applicable. 
# 2. Analogies: 
#     - Use relatable analogies to simplify complex ideas. 
# If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing. 
# Ensure the summary is clear, accurate, and aligned with the provided style and length.
# """,
# input_variables=['paper_input','style_input','length_input'],
# validate_template=True
# )
# # fill the placeholders
# prompt = template.invoke({
#     'paper_input' : paper_input,
#     'style_input' : style_input,
#     'length_input' : length_input
# })

# if st.button("Summarize"):
#     result = model.invoke(prompt)
#     st.write(result.content)