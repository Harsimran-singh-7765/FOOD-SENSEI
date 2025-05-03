import os
from dotenv import load_dotenv,find_dotenv
import google.generativeai as genai
from collections import defaultdict


load_dotenv(find_dotenv())


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


chat_sessions = {}


SYSTEM_PROMPT = (
    "You are a scientific nutrition expert AI trained to bust common food myths. "
    "When someone asks or states something possibly false about food, nutrition, or health, "
    "you respond with clear explanations backed by evidence or scientific reasoning. "
    "Keep it friendly and helpful, but correct the myth directly."
)

def get_chat_response(user_input, session_id):
    if session_id not in chat_sessions:
        
        model = genai.GenerativeModel("gemini-1.5-pro")
        chat = model.start_chat(history=[
            {"role": "user", "parts": [SYSTEM_PROMPT]},
            {"role": "model", "parts": ["Got it. Ready to bust some food myths!"]}
        ])
        chat_sessions[session_id] = chat
    else:
        chat = chat_sessions[session_id]

    # Send user message
    response = chat.send_message(user_input)
    
    return response.text
