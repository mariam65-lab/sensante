import os
from dotenv import load_dotenv
from groq import Groq
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "Tu es un assistant medical senegalais. Maximum 3 phrases."},
        {"role": "user", "content": "Quels sont les symptomes du paludisme ?"}
    ],
    max_tokens=200,
    temperature=0.3
)
print(response.choices[0].message.content)