import os
import time
import logging
from colorama import init, Fore
from dotenv import load_dotenv
from pipeline import build_graph
from config import (
    LLM_CHOICE,
    MAX_TOKENS,
    TEMPERATURE,
    TOP_P,
    FREQUENCY_PENALTY,
    TIMEOUT,
)

init(autoreset=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    load_dotenv()
    os.makedirs("./data", exist_ok=True)

    if LLM_CHOICE.lower() == "gemini":
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY:
            raise ValueError("Google API key is not set in the environment variables.")
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            frequency_penalty=FREQUENCY_PENALTY,
            timeout=TIMEOUT,
            max_retries=2,
        )
    elif LLM_CHOICE.lower() == "llama":
        from llama_utils import ChatLlama
        from config import LLAMA_MODEL  # Now available in config.py

        llm = ChatLlama(
            model=LLAMA_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            timeout=TIMEOUT,
            max_retries=2,
        )
    else:
        raise ValueError(f"Unknown LLM_CHOICE: {LLM_CHOICE}")

    logging.info(
        f"{Fore.CYAN}🚀 Starting the ESCO skill sentence generation process using {LLM_CHOICE}..."
    )
    start_time = time.perf_counter()

    graph = build_graph(llm)
    app = graph.compile()

    initial_state = {"esco_skills": [], "processed_count": 0}
    final_state = app.invoke(initial_state)

    elapsed = time.perf_counter() - start_time
    logging.info(
        f"{Fore.GREEN}🎉 Completed generation for all skills in {elapsed:.2f} seconds!"
    )


if __name__ == "__main__":
    main()
