import os
import getpass
import logging
import threading
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import ChatOllama
from pydantic import BaseModel, Field


load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Set Google AI API key. TODO: Refactor to consider other services like claude 3.7, gpt-4o, gpt-o1
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# Global semaphore to limit concurrent GPT calls
gpt_semaphore = threading.Semaphore(2)


class SkillsOutput(BaseModel):
    skills: list[str] = Field(description="List of extracted skills as strings.")


bert_model_names = {
    "ModernBERT": "answerdotai/ModernBERT-base",
    "RoBERTa": "Jean-Baptiste/roberta-large-ner-english",
}


def load_bert_extractors():
    extractors = {}
    for model_name, model_id in bert_model_names.items():
        logging.info(f"Loading {model_name} model from {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForTokenClassification.from_pretrained(model_id)
        extractor = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
        )
        extractors[model_name] = extractor
    return extractors


# GPT Models configuration
gpt_models = {
    "Gemini": ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=60,
        max_retries=2,
    ),
    "Llama-3.2": ChatOllama(
        model="llama3.2",
        max_tokens=4000,
        temperature=0.7,
        top_p=0.95,
    ),
    # "Deepseek-R1": ChatOllama(
    #     model="deepseek-r1:14b",
    #     max_tokens=4000,
    #     temperature=0.7,
    #     top_p=0.95,
    # ),
}
