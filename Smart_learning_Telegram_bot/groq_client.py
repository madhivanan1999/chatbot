from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key= os.getenv("GROQ_API_KEY"))

def ask_groq(user_id, current_prompt, history=[]):
    """Updated to handle history. 
     'history' must be a list of dictionaries.""" 
    system_instruction = ("You are a strict academic tutor. Your primary rule is: ONLY answer questions related to "
        "studies, school subjects, or academic topics. If a user asks about anything else "
        "(e.g., entertainment, casual chat, sports, personal advice), politely refuse to answer. "
        "For academic topics: "
        "1. Give a very simple explanation (beginner-friendly). "
        "2. Provide one clear example.")
    message_list = [{"role": "system", "content": system_instruction}]
    message_list.extend(history)
    message_list.append({"role": "user", "content": current_prompt})
    response = client.chat.completions.create(
        model ="llama-3.3-70b-versatile",
        messages = message_list,
        temperature = 0.5
    )
    return response.choices[0].message.content