from langchain_community.llms import Ollama
from dotenv import load_dotenv
import os


class NLPModel:
    def process_text(self, text):
        raise NotImplementedError
    

class Ollama_Model(NLPModel):
    def __init__(self, model_name):
        self.model = Ollama(model=model_name)

    def process_text(self, text):
        # Implement the logic to process text using the Ollama model
        response = self.model(text)
        return response

class GeminiAPIModel(NLPModel):
    def process_text(self, text):
        import google.generativeai as genai

        load_dotenv()

        api_key = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=api_key)
        # Set up the model
        generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 0,
        "max_output_tokens": 8192,
        }

        safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        ]

        model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                                    generation_config=generation_config,
                                    safety_settings=safety_settings)

        convo = model.start_chat(history=[
        ])
        convo.send_message(text)
        # Implement the logic to process text using the GEMINI API
        response = convo.last.text# API call to GEMINI with the text
        return response
    
def get_nlp_model(model_name=None):
    if model_name == 'gemini':
        return GeminiAPIModel()
    else:
        return Ollama_Model(model_name=model_name)