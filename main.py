with open('chiirl_events.txt', encoding='utf-8') as f:
    doc = f.read()
    text3 = doc.split('____________________')

for i in range(10):
    print(text3[i], end='\n - - - \n')

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

with open('wiki_game_awards_2025.txt', encoding='utf-8') as f:
    doc = f.read()
    text1 = text_splitter.split_text(doc)

for i in range(10):
    print(text1[i], end='\n \n - - - \n \n')


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

with open('wiki_98th_oscars.txt', encoding='utf-8') as f:
    doc = f.read()
    text2 = text_splitter.split_text(doc)

for i in range(5):
    print(text2[i], end='\n \n - - - \n \n')
    

from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

from langchain_core.documents import Document

document = Document(
    page_content='testing document'
)

vector_store.add_documents(
    documents=[document],
    ids=['test']
)

vector_store.get_by_ids(['test'])

results = vector_store.similarity_search_with_score(
    'Will it be hot tomorrow?', k=1
)

print(results)
print('score = ', results[0][1])

import math
t1 = embeddings.embed_query('testing document')
t2 = embeddings.embed_query('Will it be hot tomorrow?')


# add the 3 documents to the vector db

doc_list = []

for chunk in text1:
    doc_list.append(Document(page_content=chunk))

for chunk in text2:
    doc_list.append(Document(page_content=chunk))

for chunk in text3:
    doc_list.append(Document(page_content=chunk))

keys = vector_store.add_documents(documents=doc_list)


# try some similarity search

vector_store.similarity_search_with_score('Clair Obscur: Expedition 33 how many nominations at 2025 game awards', k=5)
vector_store.similarity_search_with_score('chicago tech meetup events dec 2025', k=5)
vector_store.similarity_search_with_score('time and location of 2026 academy awards', k=5)


# rag!

prompt = 'who performed at the 2025 game awards'

x = vector_store.similarity_search_with_score('who performed at the 2025 game awards', k=5)

rag_string = ''

for doc in x:
    y = doc[0].model_dump()['page_content']
    rag_string += y
    rag_string += '\n \n'

new_prompt = f'''

using this information:

{rag_string}

answer this question: {prompt}

'''

from google import genai

client = genai.Client(api_key='x')

response = client.models.generate_content(
    model="gemini-2.5-flash", contents=new_prompt
)

print(response.text)


# same prompt without rag

response = client.models.generate_content(
    model="gemini-2.5-flash", contents="who performed at the 2025 game awards"
)

print(response.text)