import os

# importing Langchain libraries
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain

import streamlit as st

apikey = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="New Approach")

llm_noncreative = OpenAI(temperature=0)
llm_creative = OpenAI(temperature=0.6)


st.title("New Langchain Approach")

# Define the user input

UserInput = st.text_area(label="Input Value", height=500)

Dictionary = """ 
    Effervescent, Enigmatic, Mellifluous, Surreptitious, Ephemeral, Clandestine, Serendipity,   Quintessential, Ethereal, Opulent, Melancholy, Resplendent, Nostalgia, Effusive, Labyrinthine,    Incandescent, Ineffable, Serene, Resonance, Serenity, Euphoria, Querulous, Luminary, Bucolic, Tranquil

"""


# Identify Words - Block_1
identify_words = PromptTemplate(
    input_variables=["UserInput", "Dictionary"],
    template=""" 

    You will be given two inputs, first an input paragraph, and second, a set of random words will be given. Your work is to identify the 8 best words from the set of random words that resonate the best with the paragraph and can potentially be used in context with the paragraph, as the words will be used to further help in paraphrasing the user input. Don't output explanations. Only Output the five words in a comma seperated list, don't be verbose and don't output anything that is not required.

    The Input Paragraph is : {UserInput}  
    The Set of Words is : {Dictionary}

    """,
)

# Paraphrasing using The 4R's Method - Block_2
paraphrase = PromptTemplate(
    input_variables=["UserInput", "final_words"],
    template="""
    You are a Student who will be punished if he uses any words other than the given list of words and will fail the test. Follow this instructions if you want to pass : 
    You will be given two inputs first, is a user input paragraph, secondly, a list of some words. Your work is to paraphrase the input paragraph using the 4R method, which will be explained below.
    Use all the words from the input word list that will be given. All of them. Wrap the new words with a * wherever used.

    The User Input is : {UserInput}
    The Word List is : {final_words}

    You need to paraphrase the User Input - which you need to do using the input words which will i gave to you earlier and use all of them

    Make Sure to Incorporate all the words anyhow, i mean anyhow, in the final output. Don't Assume Anything and Use any words other than the given input

    The output should be in the format : ``New Words Used`` and ``Rephrased Parahraph``
    
    Don't output any explanations, only output the final paraphrased output. Don't be verbose and don't output anything that is not required. 
""",
)


if st.button("Generate"):
    with st.spinner(text="Processing"):
        st.success("Done !")
        # Defining chains and inputs for block_1
        identify_chain = LLMChain(llm=llm_noncreative, prompt=identify_words)
        identified_words = identify_chain.run(
            {"UserInput": UserInput, "Dictionary": Dictionary}
        )

        final_words = identified_words

        # Defining chains and inputs for block_2
        paraphrase_chain = LLMChain(llm=llm_creative, prompt=paraphrase)
        paraphrased = paraphrase_chain.run(
            {"UserInput": UserInput, "final_words": final_words}
        )

        st.text_area(value=paraphrased, label="Output", height=500)
