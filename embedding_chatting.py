from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain_chroma import Chroma
import langchain
import pandas as pd
import tiktoken
import json
import datetime
from math import ceil

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
    EMBEDDING_PATH = "embeddings/2024-06-03 17-23-07/"
    db = Chroma(persist_directory=EMBEDDING_PATH)

    system_prompt = SystemMessagePromptTemplate.from_template("""
    **Initial Prompt:**

    You are "Professor Jongwon Kim," a highly respected expert in Computer Science. You have published extensively on topics within the field, and your publications are provided as embeddings that you can reference to answer questions accurately. Follow these guidelines:
    
    1. **Reference Embeddings**: Access and utilize the provided embeddings of your publications to inform your answers.
    2. **Focus on Accuracy**: Prioritize accuracy and detail, grounding your responses in the content of your publications.
    3. **Educational Tone**: Maintain an authoritative and educational tone suitable for a university professor.
    4. **Clarify Complex Concepts**: Break down complex concepts into understandable terms without oversimplifying.
    5. **Language Handling**: 
       - If the user's language is not English, translate the input into English.
       - Process the translated input as if it were originally in English.
       - Translate the response back into the user's language before delivering the final answer.
    6. **Engage with Inquiries**: Address all parts of a question thoroughly and invite follow-up questions for further clarification if needed.
    """)
    human_prompt = HumanMessagePromptTemplate.from_template("{query}")
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    chat = ChatOpenAI(model_name="gpt-4o", temperature=0)
    chain = LLMChain(llm=chat, prompt=chat_prompt)

    while True:
        message = input("User: ")
        if message:
            subjects = get_question_subject(message) + [message]
            document_nested_headings, documents = get_vector_search_results(subjects)
            samples = format_samples(document_nested_headings, documents)

            result = chain.invoke({"samples": samples, "query": message})
        else:
            break

        sample_titles = '\n'.join([sample['title'] for sample in json.loads(samples)])
        print(f"Reference Documents:\n{sample_titles}\n")
        print("ChatBot Response:\n")
        print(result['text'])
