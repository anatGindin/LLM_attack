import google.generativeai as genai
import os
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Consume the generator into a list
models = list(genai.list_models())
print(models)
