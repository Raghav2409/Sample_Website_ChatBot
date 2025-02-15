import streamlit as st
import pandas as pd
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from uuid import uuid4
import openai
import os

# Set API key
os.environ['OPENAI_API_KEY'] = 'your_openai_api_key'
# Setup embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Setup ChromaDB vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"  # Local directory to save data, remove if not needed
)
# Load and process CSV once at the start
def load_documents():
    data = pd.read_csv('urls_and_chunks.csv')
    documents = [
        Document(
            page_content=row['Chunk_Content'],
            metadata={'URL': row['URL'], 'Chunk_Index': row['Chunk_Index']},
            id=str(uuid4())
        ) for index, row in data.iterrows()
    ]
    vector_store.add_documents(documents=documents, ids=[doc.id for doc in documents])

load_documents()

def check_topic_allowed(user_request):
    if 'last_question_allowed' in st.session_state and st.session_state.last_question_allowed:
        return 'allowed'
    guardrail_prompt = """
    
    You are an intelligent filter trained to identify if questions are relevant to Apexon's services and solutions in areas like healthcare, finance, technology implementation, and media analysis.
    
    Query: "Tell me about Apexon's work in cloud solutions."
    Response: allowed
    
    Query: "What is the best dog breed for apartment living?"
    Response: not_allowed

    Query: "Does Apexon provide services in the field of petcare?"
    Response: not_allowed

    Query: "How have recent AI integrations impacted customer retention rates in the healthcare sector?"
    Response: allowed

    Query: "What are the key factors in successfully deploying large-scale data analytics platforms in financial services?"
    Response: allowed

    # Follow-up questions
    Query: "Tell me more about this success story."
    Response: allowed
    
    Query: "Can you elaborate on the reduction in dispute handling time?"
    Response: allowed
    
    
    """
    messages = [
        {"role": "system", "content": guardrail_prompt},
        {"role": "user", "content": user_request}
    ]
    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content

# Streamlit App
def app():
    st.title('Apexon Query Answer Interface')

    # Initialize session state for storing conversation history
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    user_query = st.text_area("Enter your query:", height=150)

    if st.button("Generate Answer"):
        if check_topic_allowed(user_query) == 'allowed':
            # Retrieve the context from previous conversations
            context = "\n".join([f"User: {entry['user']}\nAssistant: {entry['assistant']}" for entry in st.session_state.conversation])
            
            # Generate response with continuity
            response = generate_response(user_query, context)
            
            # Append the current conversation to the session state
            st.session_state.conversation.append({"user": user_query, "assistant": response})
        else:
            response = "Your query is not relevant to Apexon's services and solutions. Please ask another question."
            st.session_state.conversation.append({"user": user_query, "assistant": response})

    # Display the conversation history in a chatbot format
    for entry in st.session_state.conversation:
        st.markdown(f"**User:** {entry['user']}")
        st.markdown(f"**Assistant:** {entry['assistant']}")
        st.markdown("---")
def generate_response(query, context):
    results = vector_store.similarity_search(query, k=3)  # Adjust 'k' as needed
    additional_context = "\n\n".join([f"Content from {res.metadata['URL']}:\n{res.page_content}" for res in results])
    
    # Set up query for GPT endpoint
    messages = [
        {"role": "system", "content": """You are a knowledgeable, articulate, and user-friendly assistant representing Apexon. Your primary goal is to generate crisp, accurate, and contextually relevant answers based on the user’s input, specifically related to our services, solutions, and expertise at Apexon. Be concise yet informative, provide step-by-step guidance when needed, and offer clarifications to avoid misunderstandings. Maintain a professional and approachable tone that reflects our brand identity. When using data from retrieved content, generate an organized and coherent response with clear sections, linking your answers to our capabilities, case studies, and service offerings whereever needed."""},
        {"role": "user", "content": query},
        {"role": "assistant", "content": context + additional_context}
    ]
    
    # Call the GPT endpoint
    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    app()