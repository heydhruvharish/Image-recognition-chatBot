import streamlit as st
from openai import OpenAI
import base64

with open ("requirements.txt","r") as file:
    instruction=file.read()

def encode_image(uploaded_file):
    """Converts a Streamlit UploadedFile to a base64 string."""
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        return base64.b64encode(bytes_data).decode('utf-8')
    return None


st.set_page_config(page_title="Vision Chatbot", page_icon="👁️", layout="wide")
st.title("Multimodal Vision Chatbot 👁️💬")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=st.secrets["GROQ_API_KEY"]
)


if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("Upload Context")
    st.info("Upload an image here, then ask questions about it in the main chat area.")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask a question about the image..."):
    

    with st.chat_message("user"):
        st.markdown(prompt)
        

    st.session_state.messages.append({"role": "user", "content": prompt})

    api_messages = [
        {"role": "system", "content": instruction}
    ]
    
    # Add historical messages (text only for context)
    for msg in st.session_state.messages[:-1]: 
        api_messages.append({"role": msg["role"], "content": msg["content"]})
        
    # Construct the current message with image if present
    current_content = [{"type": "text", "text": prompt}]
    
    if uploaded_image:
        base64_image = encode_image(uploaded_image)
        current_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })
        
    api_messages.append({"role": "user", "content": current_content})
    
   
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            response = client.chat.completions.create(
                model="qwen/qwen3.8-27b",
                messages=api_messages,
                max_tokens=500,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
           
            message_placeholder.markdown(full_response)
            
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            message_placeholder.error(f"An error occurred: {e}")