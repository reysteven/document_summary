from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import PyPDF2

import shutil
import os
os.environ['OPENAI_API_KEY'] = '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'

def main():

    db_path = 'C:\\Users\\miku39\\Documents\\rnd\\gpt_handson\\document_summary\\db'
    if os.path.isdir(db_path):
        shutil.rmtree(db_path)

    file_sample = open("englishclub-7-secrets.pdf", 'rb')

    pdfReader = PyPDF2.PdfReader(file_sample)

    for j in range(len(pdfReader.pages)):

        print(f'Extracting PDF at page {j+1}')

        doc_sample = pdfReader.pages[j].extract_text()

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_text(doc_sample)
        print(texts)

        if len(texts) > 0:

            embeddings = OpenAIEmbeddings()

            docsearch = Chroma.from_texts(
                texts,
                embeddings,
                metadatas=[{'source': f'Page {j} - Text chunk {i} of {len(texts)}'} for i in range(len(texts))],
                persist_directory=db_path
            )

            docsearch.persist()
            docsearch = None
    
    file_sample.close()

if __name__ == '__main__':
    main()