import streamlit as st
import requests
import uuid
import os

# Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000/api/v1")

st.set_page_config(
    page_title="Production RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Controls
with st.sidebar:
    st.title("⚙️ RAG Settings")
    
    st.subheader("Model Configuration")
    model = st.selectbox(
        "Model Selector",
        ["gpt-4o", "gpt-4o-mini"]
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1
    )
    
    max_docs = st.slider(
        "Max Retrieved Docs",
        min_value=1,
        max_value=10,
        value=4,
        step=1
    )
    
    st.divider()
    
    st.subheader("📄 Upload Documents")
    uploaded_file = st.file_uploader("Upload a file to index", type=["txt", "pdf", "md"])
    if uploaded_file:
        if st.button("Ingest Document"):
            with st.spinner("Uploading and indexing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{API_BASE_URL}/ingest", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Indexed {data.get('chunks_indexed', 0)} chunks from {uploaded_file.name}")
                    else:
                        st.error(f"Failed to index: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")
                    
    st.divider()
    
    # Health check status
    try:
        health_resp = requests.get(f"{API_BASE_URL}/health")
        if health_resp.status_code == 200:
            st.success("API: Connected")
        else:
            st.warning("API: Issues detected")
    except:
        st.error("API: Disconnected. Is FastAPI running?")

# Main chat interface
st.title("🤖 Production RAG Assistant")
st.markdown("Ask questions about your indexed documents.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if available and it's an assistant message
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")
                    
        # Display metadata badges
        if message["role"] == "assistant" and message.get("latency"):
            cols = st.columns(4)
            cols[0].caption(f"⏱️ {message['latency']}ms")
            if message.get("tokens"):
                cols[1].caption(f"🪙 {message['tokens']} tokens")

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Thinking..."):
            try:
                # Call backend API
                payload = {
                    "question": prompt,
                    "session_id": st.session_state.session_id,
                    "user_id": "streamlit_user"
                }
                
                response = requests.post(f"{API_BASE_URL}/chat", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No answer received.")
                    sources = data.get("sources", [])
                    latency = data.get("latency_ms", 0)
                    tokens = data.get("token_count", 0)
                    
                    message_placeholder.markdown(answer)
                    
                    if sources:
                        with st.expander("View Sources"):
                            for source in sources:
                                st.markdown(f"- {source}")
                                
                    cols = st.columns(4)
                    cols[0].caption(f"⏱️ {latency}ms")
                    cols[1].caption(f"🪙 {tokens} tokens")
                    
                    # Save assistant message to state
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources,
                        "latency": latency,
                        "tokens": tokens
                    })
                else:
                    error_detail = response.json().get('detail', 'Unknown error')
                    st.error(f"API Error ({response.status_code}): {error_detail}")
            except Exception as e:
                st.error(f"Connection failed: {str(e)}. Make sure FastAPI is running on port 8000.")
