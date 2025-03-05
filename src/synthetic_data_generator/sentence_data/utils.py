import os
import json
import logging
import threading
from colorama import Fore


def update_json_file(file_path: str, new_entry: dict, lock: threading.Lock) -> None:
    """
    Safely update (or create) a JSON file by appending new_entry.
    """
    with lock:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                except Exception:
                    data = []
        else:
            data = []
        data.append(new_entry)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
    logging.info(
        f"{Fore.GREEN}Updated {file_path} with new entry for ESCOID: {new_entry['escoid']}"
    )


def generate_prompt_text(data_type: str, skill: dict) -> str:
    """
    Create a prompt for Gemini that asks for 20 sentences about the skill,
    where 12 explicitly mention the skill and 8 implicitly mention it.
    """
    explicit_instructions = "12 sentences must explicitly mention the skill."
    implicit_instructions = "8 sentences should mention the skill implicitly."
    prompt = (
        f"Using the following skill information:\n"
        f"ESCOID: {skill['conceptUri']}\n"
        f"Preferred Label: {skill['preferredLabel']}\n"
        f"Description: {skill['description']}\n\n"
        f"Generate 20 {data_type} sentences. {explicit_instructions} {implicit_instructions}\n"
        f"Return a valid JSON object with the following structure:\n"
        f'{{"escoid": "<ESCOID>", "preferredLabel": "<Preferred Label>", "description": "<Description>", '
        f'"sentences": {{"explicit": [<list of 12 sentences>], "implicit": [<list of 8 sentences>]}}}}\n'
        f"Do not include any markdown formatting or extra text."
    )
    return prompt


def call_llm_for_skill(data_type: str, skill: dict, llm) -> dict:
    """
    Calls the LLM (Gemini) with a prompt for the given data type (course, job, or cv)
    and returns the parsed JSON output.
    """
    prompt_text = generate_prompt_text(data_type, skill)
    logging.info(
        f"{Fore.MAGENTA}Requesting {data_type} sentences for skill '{skill['preferredLabel']}' (ESCOID: {skill['conceptUri']})"
    )
    result = llm.invoke(
        prompt_text
    )  # Assuming a simple interface like llm.invoke(prompt_text)
    output_text = result.content if hasattr(result, "content") else str(result)
    try:
        parsed = json.loads(output_text)
    except Exception as e:
        raise ValueError(
            f"Failed to parse JSON from LLM for {data_type} on skill {skill['preferredLabel']}: {e}. Raw output: {output_text}"
        )
    return parsed
