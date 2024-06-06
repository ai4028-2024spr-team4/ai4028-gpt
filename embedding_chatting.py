import os

from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain_chroma import Chroma
import langchain
import pandas as pd
import tiktoken
import json
import datetime
from math import ceil

from langchain import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

if False:
    EMBEDDING_PATH = f"embeddings/{datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}/"
    rows = []
    with open(f"publications_txt/filtered.json", "r") as f:
        filter_result = json.load(f)
    for file in filter_result:
        file: str
        if filter_result[file]["result"] is False:
            continue
        with open(f'publications_txt/{file}', 'r') as f:
            text = f.read()
            token_cnt = len(tiktoken.get_encoding('cl100k_base').encode(text))
            if token_cnt > 8000:
                splits  = ceil( token_cnt / 7500 )
                tokens = tiktoken.get_encoding('cl100k_base').encode(text)
                texts = []
                for i in range(splits):
                    texts.append(tiktoken.get_encoding('cl100k_base').decode(tokens[i*7500:(i+1)*8000]))
            else:
                texts = [text]
            for text in texts:
                rows.append((file.replace(".txt", ""), text))
    df = pd.DataFrame(rows, columns=["Embedding Target", "Document"])

    oai_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    loader = DataFrameLoader(df, page_content_column="Embedding Target")
    db = Chroma.from_documents(loader.load(), embedding=oai_embeddings, persist_directory=EMBEDDING_PATH)

if True:
    system_prompt = SystemMessagePromptTemplate.from_template("""
    **Initial Prompt:** You are "Professor Jongwon Kim," a distinguished Computer Science expert with extensive publications provided in this prompt. Your responses should adhere to the following guidelines:
    1. **Reference Publications**: Use the provided publications to inform your answers.
    2. **Focus on Accuracy**: Ensure all answers are detailed and accurate, grounding responses in the content of your publications.
    3. **Educational Tone**: Maintain an authoritative, educational tone expected from a university professor.
    4. **Clarify Concepts**: Simplify complex concepts without oversimplifying.
    5. **Language Handling**: 
       - Translate non-English input to English.
       - Process input as if originally in English.
       - Translate responses back to the user's language.
    6. **Engage with Inquiries**: Address all question components thoroughly and invite follow-up questions.
    
    **Your Publications**
    All publications are given as title: abstract pair.
    {publications}
    """)
    human_prompt = HumanMessagePromptTemplate.from_template("{query}")
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    chat = ChatOpenAI(model_name="gpt-4o", temperature=0)
    chain = LLMChain(llm=chat, prompt=chat_prompt)

    with open('publications_summarized.txt', 'r') as f:
        publications = f.read()

    while True:
        message = input("User: ")
        if message:
            result = chain.invoke({"publications": publications, "query": message})
        else:
            break

        print("ChatBot Response:\n")
        print(result['text'])
