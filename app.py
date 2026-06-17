import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 🚨 THE PATH FIX: Force Python to find the .env file in its own exact folder
current_directory = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_directory, ".env")
load_dotenv(dotenv_path=env_path)

# Verify the environment variable was found locally before initializing
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("⚠️ WARNING: GEMINI_API_KEY could not be read from the .env file!")

# Initialize the client using the environment variable
client = genai.Client(api_key=api_key)

app = FastAPI(title="Medical Translation Engine (Free / Non-Device)")
# The CORS Fix: Allows your index.html website to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranslationRequest(BaseModel):
    complex_text: str


@app.post("/translate")
async def translate_medical_text(request: TranslationRequest):
    if not request.complex_text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        config = types.GenerateContentConfig(
            temperature=0.0,
            system_instruction=(
                "You are an administrative and educational language translation tool. "
                "Your ONLY job is to change high-level clinical vocabulary into clear, "
                "patient-friendly language suitable for a 6th-grade reading level. "
                "REGULATORY SAFETY CRITERIA: You must NOT alter clinical intent, you must NOT diagnose, "
                "and you must NOT suggest treatments, alternatives, or clinical next steps."
            )
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Translate the following medical text: {request.complex_text}",
            config=config
        )

        return {"translated_text": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Engine Runtime Error: {str(e)}")