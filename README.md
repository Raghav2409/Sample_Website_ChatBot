# Streamlit Chatbot with Conversation Memory

This repository contains code for a Streamlit-based chatbot application that maintains short-term memory of conversation history. The chatbot is designed to answer queries related to Apexon's services and solutions, leveraging vector search and OpenAI's language models.

## Conversation Memory Implementation

The application implements short-term memory in the chat through several key components:

### 1. Session State for Conversation History

```python
# Initialize session state for storing conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
```

Streamlit's session state is used to maintain a list of all user queries and assistant responses throughout the session.

### 2. Appending New Exchanges

```python
# Append the current conversation to the session state
st.session_state.conversation.append({"user": user_query, "assistant": response})
```

Each new interaction between the user and assistant is stored as a dictionary with 'user' and 'assistant' keys in the conversation list.

### 3. Context Retrieval for Continuity

```python
# Retrieve the context from previous conversations
context = "\n".join([f"User: {entry['user']}\nAssistant: {entry['assistant']}" for entry in st.session_state.conversation])
```

When generating a new response, the application formats all previous exchanges as a single string to provide conversation context.

### 4. Context Utilization in Response Generation

```python
messages = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": query},
    {"role": "assistant", "content": context + additional_context}
]
```

The assembled context is passed to the language model, allowing it to generate responses that maintain continuity with the ongoing conversation.

## Key Features

- Maintains conversation history within a session
- Provides contextual awareness to the language model
- Combines conversation history with relevant vector search results
- Topic filtering to ensure queries are relevant to Apexon's domain

## Dependencies

- Streamlit
- Pandas
- LangChain (with Chroma)
- OpenAI API
- UUID

## Usage

1. Set your OpenAI API key
2. Prepare a CSV file with your document chunks
3. Run the application with `streamlit run app.py`

## Limitations

- Memory persists only for the duration of the session
- No long-term memory or user profile persistence
- Context window is limited by the token limits of the underlying language model

## Future Improvements for Farm AI

- Implement database storage for long-term memory
- Add user authentication for personalized conversation history
- Implement memory summarization for handling very long conversations
- Add priority weighting to more recent conversations
