from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
import streamlit as st 
from langchain_core.prompts import PromptTemplate, load_prompt

load_dotenv()

@st.cache_resource
def load_model():
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        model_kwargs={"cache_dir": "../hf_cache"},
        pipeline_kwargs={
            "temperature": 0.1,
            # "max_new_tokens": 500,
            "return_full_text": False,
        }
    )
    return ChatHuggingFace(llm=llm)

model = load_model()

st.header('Research Tool')

paper_input = st.selectbox( "Select Research Paper Name :", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style :", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length :", ["Short (1 paragraph)", "Medium (3 paragraphs)", "Long (5 paragraphs)"] )


template = load_prompt('template.json')

# fill the placeholders
# prompt = template.invoke({
#     'paper_input' : paper_input,
#     'style_input' : style_input,
#     'length_input' : length_input
# })

if st.button("Summarize"):
    chain = template | model
    result = chain.invoke({
        'paper_input' : paper_input,
        'style_input' : style_input,
        'length_input' : length_input
    })

    # result = model.invoke(prompt)
    st.write(result.content)