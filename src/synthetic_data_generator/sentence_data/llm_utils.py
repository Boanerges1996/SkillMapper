import json
import logging
import time
import threading
from colorama import Fore
from config import MAX_RETRIES
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


# Define the structured output schema using Pydantic.
class SentenceOutput(BaseModel):
    escoid: str = Field(..., description="The full ESCO URL")
    preferredLabel: str
    description: str
    sentences: dict = Field(
        ...,
        example={
            "explicit": [
                "Example explicit sentence 1.",
                "Example explicit sentence 2.",
                # ... total 12 explicit sentences expected.
            ],
            "implicit": [
                "Example implicit sentence 1.",
                "Example implicit sentence 2.",
                # ... total 8 implicit sentences expected.
            ],
        },
    )


# Create the structured output parser.
output_parser = PydanticOutputParser(pydantic_object=SentenceOutput)


def update_json_file(file_path: str, new_entry: dict, lock: threading.Lock) -> None:
    """
    Safely update (or create) a JSON file by appending new_entry.
    """
    import os  # local import for clarity

    with lock:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except Exception:
            data = []
        data.append(new_entry)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
    logging.info(
        f"{Fore.GREEN}Updated {file_path} with new entry for ESCOID: {new_entry.get('escoid', 'N/A')}"
    )


def generate_prompt_text(data_type: str, skill: dict) -> str:
    """
    Create a prompt that instructs Gemini to generate 20 sentences
    (12 explicit and 8 implicit) in a valid JSON format matching SentenceOutput.
    """
    format_instructions = output_parser.get_format_instructions()
    example_output = {
        "escoid": skill["conceptUri"],
        "preferredLabel": skill["preferredLabel"],
        "description": skill["description"],
        "sentences": {
            "explicit": [
                "Example explicit sentence 1.",
                "Example explicit sentence 2.",
                # ... up to 12 sentences
            ],
            "implicit": [
                "Example implicit sentence 1.",
                "Example implicit sentence 2.",
                # ... up to 8 sentences
            ],
        },
    }
    prompt = (
        f"Using the following skill information:\n"
        f"ESCOID: {skill['conceptUri']}\n"
        f"Preferred Label: {skill['preferredLabel']}\n"
        f"Description: {skill['description']}\n\n"
        f"Generate 20 {data_type} sentences, with 12 sentences explicitly mentioning the skill and 8 sentences implicitly mentioning it.\n"
        f"Output ONLY a valid JSON object that exactly matches the following format (do not include any extra text, markdown, or explanation):\n\n"
        f"{format_instructions}\n\n"
        f"Example output:\n"
        f"{json.dumps(example_output, indent=2)}\n\n"
        f"Ensure that the 'escoid' field contains the full ESCO URL.\n"
        f"Now, generate the JSON output."
    )
    return prompt


def clean_json_output(raw: str) -> str:
    """
    Cleans the raw LLM output by removing markdown code fences if present.
    If the output starts with "```json" or "```", it will remove the first and last lines.
    """
    raw = raw.strip()
    if raw.startswith("```json"):
        lines = raw.splitlines()
        # Remove the first line (```json) and the last line if it is ```
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    elif raw.startswith("```") and raw.endswith("```"):
        # Fallback if not specifically "```json"
        raw = raw.strip("`").strip()
    return raw


def fix_output_json(parsed: dict, skill: dict) -> dict:
    """
    Ensures that the output JSON uses the key 'escoid' (all lowercase)
    and that its value is the full URL from the skill data.
    """
    full_url = skill.get("conceptUri", "")
    if "escoId" in parsed:
        parsed["escoid"] = parsed.pop("escoId")
    if "escoid" in parsed:
        if not parsed["escoid"].startswith("http") or len(parsed["escoid"]) < len(
            full_url
        ):
            parsed["escoid"] = full_url
    else:
        parsed["escoid"] = full_url
    return parsed


def call_llm_for_skill(data_type: str, skill: dict, llm) -> dict:
    """
    Calls the LLM with a prompt for the given data_type (course, job, or cv)
    and returns the parsed JSON output. Uses exponential backoff on failures.
    """
    prompt_text = generate_prompt_text(data_type, skill)
    logging.info(
        f"{Fore.MAGENTA}Requesting {data_type} sentences for skill '{skill['preferredLabel']}' (ESCOID: {skill['conceptUri']})"
    )

    attempt = 0
    delay = 2
    while attempt < MAX_RETRIES:
        try:
            result = llm.invoke(prompt_text)
            output_text = result.content if hasattr(result, "content") else str(result)
            output_text = clean_json_output(output_text)
            if not output_text:
                raise ValueError("LLM returned an empty output.")
            parsed = json.loads(output_text)
            parsed = fix_output_json(parsed, skill)
            # Validate the parsed output against the schema.
            SentenceOutput.parse_obj(parsed)
            return parsed
        except Exception as e:
            attempt += 1
            logging.warning(
                f"{Fore.YELLOW}Attempt {attempt}/{MAX_RETRIES} failed for {data_type} on skill {skill['preferredLabel']}: {e}"
            )
            time.sleep(delay)
            delay *= 2
    raise ValueError(
        f"Failed to get valid JSON for {data_type} on skill {skill['preferredLabel']} after {MAX_RETRIES} attempts"
    )
