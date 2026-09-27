# Multimodal Vision Chatbot 👁️💬

An interactive conversational AI chatbot that can see. Built with Python and Streamlit, this application uses Groq's high-speed inference API and vision-language models (VLMs) to process user-uploaded images and answer natural language questions about their contents.

## Features
* **Multimodal Chat:** Upload images (JPG, PNG) and ask questions about them in a ChatGPT-like interface.
* **Context-Aware:** The chatbot remembers conversation history for natural follow-up questions.
* **High-Speed Inference:** Powered by Groq's ultra-fast API using the `qwen/qwen3.8-27b` vision model.
* **Secure API Key Management:** Uses Streamlit's native secrets management to keep credentials safe.

## Tech Stack
* **Frontend:** Streamlit
* **API Client:** OpenAI Python SDK (configured for Groq)
* **Image Processing:** Pillow (PIL)

---


