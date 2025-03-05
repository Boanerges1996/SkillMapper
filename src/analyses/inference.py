import re
import json
import time
import logging
from config import gpt_semaphore, SkillsOutput


def bert_extract_skills_from_pipeline(text: str, extractor):
    """
    Use the given BERT NER pipeline to extract skills from text.
    Performs basic cleaning (removes punctuation) and deduplication.
    """
    logging.info("Starting BERT extraction using extractor...")
    start_time = time.time()
    ner_results = extractor(text)
    extracted = [
        entity["word"].lower() for entity in ner_results if len(entity["word"]) > 2
    ]
    extracted = list(
        set([re.sub(r"[^a-zA-Z0-9\s]", "", skill).strip() for skill in extracted])
    )
    elapsed = time.time() - start_time
    logging.info(
        f"BERT extraction complete in {elapsed:.2f} seconds; found {len(extracted)} skills."
    )
    return extracted


def gpt_extract_skills_from_model(
    text: str, gpt_model_instance, model_name: str, max_attempts=5, base_delay=5
):
    """
    Use the given GPT model instance (via LangChain) to extract skills.
    For ChatOllama models (e.g. Llama-3.2 and Deepseek-R1), use plain invocation with manual JSON parsing.
    For models like Gemini, use structured output.
    """
    logging.info(f"Starting GPT extraction using {model_name}...")
    if model_name.lower() == "deepseek-r1":
        prompt = (
            "Extract all explicitly mentioned skills from the following text. "
            "Return ONLY a JSON object exactly in the following format: "
            '{"skills": ["skill1", "skill2", ...]}. '
            "Do not include any chain-of-thought, explanations, or extra text. "
            f"Text: '''{text}'''"
        )
    else:
        prompt = (
            "Extract all explicitly mentioned skills from the following text and return a JSON object "
            "with a key 'skills' mapping to a list of strings. Do not include any additional text. "
            f"Text: '''{text}'''"
        )

    attempt = 1
    while attempt <= max_attempts:
        try:
            with gpt_semaphore:
                start_time = time.time()
                if model_name.lower() in ["llama-3.2", "deepseek-r1"]:
                    response = gpt_model_instance.invoke(prompt).content.strip()
                    if not response:
                        raise ValueError("Empty response received.")
                    logging.info(f"{model_name} raw response: {response}")
                    try:
                        parsed = json.loads(response)
                        skills = (
                            parsed["skills"]
                            if isinstance(parsed, dict) and "skills" in parsed
                            else []
                        )
                    except Exception as parse_error:
                        logging.warning(
                            f"Manual JSON parsing failed for {model_name} on attempt {attempt}: {parse_error}. Response: {response}"
                        )
                        raise ValueError("JSON parsing error")
                else:
                    model_with_structure = gpt_model_instance.with_structured_output(
                        SkillsOutput
                    )
                    structured_output = model_with_structure.invoke(prompt)
                    skills = structured_output.skills
                elapsed = time.time() - start_time
                logging.info(
                    f"{model_name} extraction complete in {elapsed:.2f} seconds on attempt {attempt}."
                )
                return [skill.lower() for skill in skills]
        except Exception as e:
            logging.error(f"{model_name} extraction failed on attempt {attempt}: {e}")
            delay = base_delay * attempt
            logging.info(
                f"Waiting {delay} seconds before retrying (attempt {attempt+1}/{max_attempts})."
            )
            time.sleep(delay)
            attempt += 1
    logging.error(f"{model_name} extraction failed after maximum attempts.")
    return []
