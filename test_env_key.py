import os
from dotenv import load_dotenv
from google import genai

# Manually load .env and print the key found (masked)
load_dotenv(override=True)
key = os.getenv("GEMINI_API_KEY")
print(f"Using key starting with: {key[:5]}...")

client = genai.Client(api_key=key)

try:
    for model in client.models.list():
        print("Success!")
        break
except Exception as e:
    print(f"Error: {e}")
