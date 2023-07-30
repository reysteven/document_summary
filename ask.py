import openai
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

import pandas as pd

import os
os.environ['OPENAI_API_KEY'] = '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'

# openai.organization = "org-uzXTgeXfczCdu5CawuJ92HEI"
# openai.api_key = "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"

def main():

    embeddings = OpenAIEmbeddings()

    docsearch = Chroma(persist_directory='db', embedding_function=embeddings)

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        OpenAI(temperature=1),
        chain_type='stuff',
        retriever=docsearch.as_retriever()
    )

    while True:

        user_input = input("What's your question: ")

        if user_input.lower() == 'finish':
            break

        # answering question based on vector DB
        result = chain({'question':user_input}, return_only_outputs=True)
        print('Answer: ' + result['answer'].replace('\n', ' '))
        print('Source: ' + result['sources'])

        # # anwering question based on general knowledge of chatGPT only
        # response = openai.ChatCompletion.create(
        #     model="gpt-3.5-turbo",
        #     messages=[
        #         {'role':'user', 'content':user_input}
        #     ]
        # )
        # print('Answer: ' + response['choices'][0]['message']['content'])

        # curr_conv_history = user_input

        # # adding conversation history to chroma db
        # text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        # texts = text_splitter.split_text(curr_conv_history)

        # if len(texts) > 0:

        #     embeddings = OpenAIEmbeddings()

        #     docsearch = Chroma.from_texts(
        #         texts,
        #         embeddings,
        #         metadatas=[{'source': f'Conversation at {pd.to_datetime("today").strftime("%Y-%m-%d %H:%M:%S")}'} for i in range(len(texts))],
        #         persist_directory='db'
        #     )

        #     docsearch.persist()
        #     docsearch = None

if __name__ == '__main__':
    main()