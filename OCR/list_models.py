"""
List available Gemini models
"""
import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY not set")
    exit(1)

genai.configure(api_key=api_key)

print("Available Gemini models:\n")
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"✓ {model.name}")
        print(f"   Display name: {model.display_name}")
        print(f"   Description: {model.description[:100]}...")
        print()
