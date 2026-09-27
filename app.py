import streamlit as st
from openai import OpenAI
import base64

# ==========================================
# 1. SYSTEM INSTRUCTION
# ==========================================
SYSTEM_INSTRUCTION = """
You are an expert multimodal visual assistant. Your goal is to provide intelligent, direct, and conversational answers based on the user's uploaded image and prompt.

CRITICAL FORMATTING RULES:
1. DIRECT ANSWERS FIRST: Always answer the user's specific question immediately. Do not start by summarizing or describing the whole image unless explicitly asked to do so.
2. NO RAW GRID/LAYOUT BREAKDOWNS: Never list spatial quadrants, cropped panels, or layout boundaries (e.g., strictly avoid writing "Top-Left:", "Bottom-Right:", "Middle-Crop:", or "Panel 1:"). Synthesize visual details naturally.
3. CONCISE & ACTIONABLE: Keep explanations clear, well-structured, and easy to read using natural bullet points or short paragraphs.
4. UNCERTAINTY: If a detail in the image is blurry, cropped, or ambiguous, state what you can see clearly and mention limitations naturally.
"""

# ==========================================
# 2. ROUND-ROBIN KEY MANAGEMENT (FIXED)
# ==========================================
# Extract all keys matching 'GROQ_API_KEY*' from st.secrets
GROQ_KEYS = [
    val for key, val in st.secrets.items() 
    if key.startswith("GROQ_API_KEY")
]

if not GROQ_KEYS:
    st.error("No Groq API keys found in st.secrets!")
    st.stop()

# Persistent state counter for round-robin rotation
if "key_index" not in st.session_state:
    st.session_state.key_index = 0

def get_client_for_attempt(attempt_offset: int) -> tuple[OpenAI, str]:
    """
    Returns an OpenAI client configured for Groq using the key index offset by the attempt count.
    Ensures that all available keys (Key 1, Key 2, Key 3...) cycle cleanly.
    """
    total_keys = len(GROQ_KEYS)
    actual_index = (st.session_state.key_index + attempt_offset) % total_keys
    selected_key = GROQ_KEYS[actual_index]
    
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=selected_key
    )
    return client, selected_key

def encode_image(uploaded_file):
    """Encodes a Streamlit UploadedFile to a Base64 string."""
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        return base64.b64encode(bytes_data).decode('utf-8')
    return None

# ==========================================
# 3. STREAMLIT UI SETUP
# ==========================================
st.set_page_config(page_title="Vision Chatbot", page_icon="👁️", layout="wide")
st.title("Multimodal Vision Chatbot 👁️💬")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Controls
with st.sidebar:
    st.header("Upload Context")
    st.info("Upload an image here, then ask questions about it in the main chat area.")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Render past chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 4. CHAT INPUT & EXECUTION
# ==========================================
if prompt := st.chat_input("Ask a question about the image..."):
    # Display user query
    with st.chat_message("user"):
        st.markdown(prompt)

    # Store text string in session state to keep message payload lightweight
    st.session_state.messages.append({"role": "user", "content": prompt})

    # System instruction
    api_messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION}
    ]
    
    # Append historical messages (Text-only to prevent sending duplicate images)
    for msg in st.session_state.messages[:-1]: 
        content = msg["content"]
        if isinstance(content, list):
            content = next((item["text"] for item in content if item.get("type") == "text"), "")
        api_messages.append({"role": msg["role"], "content": content})
        
    # Build active turn content (Attach base64 image ONLY to the current turn)
    current_content = [{"type": "text", "text": prompt}]
    
    if uploaded_image:
        base64_image = encode_image(uploaded_image)
        current_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })
        
    api_messages.append({"role": "user", "content": current_content})
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        attempts = 0
        max_attempts = len(GROQ_KEYS)
        
        while attempts < max_attempts:
            # Selected client dynamically using index + attempt count offset
            client, key_used = get_client_for_attempt(attempts)
            full_response = ""
            
            try:
                response = client.chat.completions.create(
                    model="qwen/qwen3.8-27b",
                    messages=api_messages,
                    max_tokens=600,
                    stream=True
                )
                
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
                # Advance base rotation counter for the NEXT turn
                st.session_state.key_index = (st.session_state.key_index + attempts + 1) % max_attempts
                break  # Success, exit retry loop
                
            except Exception as e:
                attempts += 1
                error_msg = str(e)
                
                # Rate limit fallback across all 3 keys
                if ("429" in error_msg or "rate_limit" in error_msg.lower()) and attempts < max_attempts:
                    message_placeholder.warning(f"Rate limit hit on key {attempts}/{max_attempts}. Rotating to next API key...")
                    continue
                else:
                    message_placeholder.error(f"An error occurred: {e}")
                    # Advance key index even on failure so the next query uses a fresh key
                    st.session_state.key_index = (st.session_state.key_index + 1) % max_attempts
                    break