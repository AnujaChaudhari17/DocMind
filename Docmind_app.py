import streamlit as st
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
import pypdf

# --- PAGE CONFIGURATION ---

st.set_page_config(page_title="TI Policy Assistant", page_icon="🧠")

# --- 1. SETUP & CACHING ---
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def process_text(text, filename):
    paragraphs = text.split('\n\n')
    chunks = []
    for i, p in enumerate(paragraphs):
        if len(p.strip()) > 20:
            chunks.append({
                "title": f"{filename} - Section {i+1}",
                "content": p.strip()
            })
    return chunks

def process_file(uploaded_file):
    text = ""
    try:
        if uploaded_file.name.endswith(".pdf"):
            reader = pypdf.PdfReader(uploaded_file)
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
        else:
            text = uploaded_file.read().decode("utf-8")
        return process_text(text, uploaded_file.name)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return []

# Function to calculate similarity
def find_best_match(query, documents, embed_model):
    if not documents:
        return None
    query_embed = embed_model.encode(query)
    best_doc = None
    highest_score = -1
    for doc in documents:
        doc_embed = embed_model.encode(doc["content"])
        score = np.dot(query_embed, doc_embed)
        if score > highest_score:
            highest_score = score
            best_doc = doc
    return best_doc

# --- NEW FUNCTION: CONTEXTUAL REWRITING ---
def rewrite_query(user_input, chat_history, client):
    """
    Uses the AI to rewrite the user's question based on history.
    Example: "What about hotels?" -> "What is the travel policy for hotels?"
    """
    if not chat_history:
        return user_input # No history, just use the raw question

    # Create a history string
    history_str = ""
    for msg in chat_history[-2:]:
        history_str += f"{msg['role']}: {msg['content']}\n"

    system_prompt = """
    Rewrite the following question to be self-contained based on the history. 
    If the question is already clear, return it as is. 
    Do NOT answer the question, just rewrite it for a search engine.
    """
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"History:\n{history_str}\n\nQuestion: {user_input}"}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.3
    )
    
    return response.choices[0].message.content

# --- 2. THE UI ---
st.title("🧠 DocuMind: Chat with your Documents")
st.markdown("Context-Aware RAG with **Groq**")

with st.sidebar:
    api_key = st.text_input("Enter Groq API Key:", type="password")
    uploaded_files = st.file_uploader("Upload Policies", type=["pdf", "txt"], accept_multiple_files=True)
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    if not api_key:
        st.warning("⚠️ Enter Groq Key to start.")
        st.stop()

client = Groq(api_key=api_key)
embed_model = get_embedding_model()

documents = []
if uploaded_files:
    with st.spinner("Processing Documents..."):
        for file in uploaded_files:
            docs = process_file(file)
            documents.extend(docs)
    st.sidebar.success(f"Loaded {len(documents)} text chunks!")
else:
    documents = [
        {"title": "Demo: Travel", "content": "Meals covered up to $50/day. Hotels up to $200/night."},
    ]

# --- 3. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the uploaded files..."):
    st.chat_message("user").markdown(prompt)
    
    # 1. Reformulate the Query (The "Memory" Step)
    with st.status("Thinking...") as status:
        status.write("Understanding context...")
        
        # We send the RAW prompt + History to Groq to get a "Search Query"
        search_query = rewrite_query(prompt, st.session_state.messages, client)
        status.write(f"Searching for: *'{search_query}'*")
        
        # 2. Search using the REWRITTEN query
        best_match = find_best_match(search_query, documents, embed_model)
        
        if best_match:
            status.write("Found relevant info!")
            context_text = f"Source: {best_match['title']}\nContent: {best_match['content']}"
            
            # 3. Generate Answer (Using Original History for flow)
            # We add the retrieved context to the system prompt
            system_msg = f"""
            You are a helpful assistant. Use the following context to answer the user's question.
            
            CONTEXT:
            {context_text}
            """
            
            # Prepare full message history for the final answer
            full_messages = [{"role": "system", "content": system_msg}]
            # Add previous chat history
            for msg in st.session_state.messages:
                full_messages.append(msg)
            # Add current user prompt
            full_messages.append({"role": "user", "content": prompt})
            
            chat_completion = client.chat.completions.create(
                messages=full_messages,
                model="llama-3.3-70b-versatile",
                temperature=0.5,
            )
            
            response_text = chat_completion.choices[0].message.content
            
            # Save to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            st.chat_message("assistant").markdown(response_text)
            status.update(label="Complete!", state="complete", expanded=False)
            
        else:
            st.warning("I couldn't find information about that in the documents.")
            status.update(label="No info found", state="error")