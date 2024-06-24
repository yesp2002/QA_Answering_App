#app.py

import streamlit as st
import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Pinecone
from transformers import pipeline
from pinecone import Pinecone, ServerlessSpec

st.title("PDF Text Search")

uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])
if uploaded_files:
    for uploaded_file in uploaded_files:
        with open(os.path.join("data", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success("Files uploaded successfully.")


question = st.text_input("Enter your question")
if question:
  # Initialize Pinecone
  pc = Pinecone(
          api_key='42000f00-9e25-43c9-9b6c-6e08ae7f9b26'
      )

  # pinecone.init(api_key='42000f00-9e25-43c9-9b6c-6e08ae7f9b26')

  # Create or connect to an existing index
  index_name = 'pdf-embeddings'
  if index_name != pc.list_indexes()[0]['name']:
      pc.create_index(index_name, dimension=384, spec=ServerlessSpec(
                  cloud='aws',
                  region='us-east-1'
              ))  # 384 is the dimension for 'all-MiniLM-L6-v2' embeddings
  index = pc.Index(index_name)

  # Load PDF and extract text
  def load_pdfs(pdf_folder):
      documents = []
      for pdf_file in os.listdir(pdf_folder):
          if pdf_file.endswith('.pdf'):
              loader = PyMuPDFLoader(os.path.join(pdf_folder, pdf_file))
              documents.extend(loader.load())
      return documents

  # Split text into chunks
  def split_text(documents):
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
      return text_splitter.split_documents(documents)

  pdf_folder = 'data'
  documents = load_pdfs(pdf_folder)
  chunks = split_text(documents)

  # Generate embeddings
  embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

  # Store embeddings in Pinecone
  # vector_store = Pinecone(api_key='42000f00-9e25-43c9-9b6c-6e08ae7f9b26', index=index, embedding=embedding_model)

  # Add chunks to Pinecone
  def add_chunks_to_pinecone(chunks):
      for i, chunk in enumerate(chunks):
          embedding = embedding_model.encode(chunk.page_content).tolist()
          metadata = {'pdf_file': chunk.metadata['source'], 'paragraph': chunk.page_content}
          index.upsert([(str(i), embedding, metadata)])

  add_chunks_to_pinecone(chunks)

  # Initialize QA pipeline
  qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

  def query_vector_database(question, top_k=5):
    query_embedding = embedding_model.encode(question).tolist()
    results = index.query(vector = query_embedding, top_k=top_k, include_metadata=True)
    return results

  # Extract the answer from the relevant paragraph
  def extract_answer(question, paragraph):
      qa_input = {
          'question': question,
          'context': paragraph
      }
      answer = qa_pipeline(qa_input)
      return answer['answer']

  def combine_chunks(chunks, index, range=1):
    combined_context = chunks[index].page_content
    if index - range >= 0:
        combined_context = chunks[index - range].page_content + " " + combined_context
    if index + range < len(chunks):
        combined_context = combined_context + " " + chunks[index + range].page_content
    return combined_context

  # Example query
  results = query_vector_database(question)

  # for result in results['matches']:
  #     st.text(f"Similarity: {result['score']}")
  #     st.text(f"PDF File: {result['metadata']['pdf_file']}")
  #     st.text(f"Paragraph: {result['metadata']['paragraph']}\n")


  if results['matches']:
    top_result = results['matches'][0]
    top_chunk_index = int(top_result['id'])
    context = combine_chunks(chunks, top_chunk_index)
    answer = extract_answer(question, context)

    st.text(f"Answer: {answer}")
    st.text(f"Source Paragraph: {context}")

  # Get the answer from the most relevant paragraph
  top_result = results['matches'][0]
  paragraph = top_result['metadata']['paragraph']
  answer = extract_answer(question, paragraph)

  st.text(f"Answer: {answer}")

  # Delete all embeddings
  def delete_all_embeddings():
      index.delete(delete_all=True)
      st.text(f"All embeddings in the '{index_name}' index have been deleted.")

  delete_all_embeddings()
