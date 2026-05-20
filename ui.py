import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="IoT Recommendation Chat")

st.title("IoT Device Recommendation Chat ")
st.write("Ask me for IoT or Smart Home device recommendations")

# Initialize session state for chat_id
if "chat_id" not in st.session_state:
    try:
        response = requests.post(f"{API_URL}/chat/create")
        response.raise_for_status()
        st.session_state.chat_id = response.json().get("chat_id")
        st.session_state.messages = []
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the backend: {e}")
        st.stop()

# Load history from backend if available
if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    try:
        history_res = requests.get(f"{API_URL}/chat/{st.session_state.chat_id}/history")
        if history_res.status_code == 200:
            backend_history = history_res.json().get("history", [])
            st.session_state.messages = []
            for msg in backend_history:
                # Map roles correctly. Backend uses "user" and "AI"
                role = "user" if msg["role"].lower() == "user" else "assistant"
                st.session_state.messages.append({"role": role, "content": msg["content"]})
    except requests.exceptions.RequestException:
        # Fine if we can't get history or backend isn't ready
        pass

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask about IoT devices..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Send to backend
    with st.chat_message("assistant"):
        with st.spinner("Analyzing request"):
            try:
                res = requests.post(
                    f"{API_URL}/chat/send/full_flow",
                    json={"chat_id": st.session_state.chat_id, "message": prompt,"email":"eyad@example.com"}
                )
                res.raise_for_status()
                print(res.json())
                response_text = res.json().get("response", "No response.")
                st.markdown(response_text)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except requests.exceptions.RequestException as e:
                error_msg = f"Error communicating with backend: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
