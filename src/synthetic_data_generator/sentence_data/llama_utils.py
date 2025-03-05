import time
import logging
from colorama import Fore
from langchain.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from config import LLAMA_MODEL


class ChatLlama:
    def __init__(
        self,
        model=LLAMA_MODEL,
        max_tokens=20000,
        temperature=0.7,
        top_p=0.9,
        timeout=30,
        max_retries=3,
    ):
        self.llm = ChatOllama(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        self.max_retries = max_retries
        self.timeout = timeout

    def invoke(self, prompt: str):
        messages = [HumanMessage(content=prompt)]
        logging.info(
            f"{Fore.CYAN}Invoking Llama 3.1 via ChatOllama with prompt (first 100 chars): {prompt[:100]}..."
        )
        attempt = 0
        delay = 2
        while attempt < self.max_retries:
            try:
                output_message = self.llm.invoke(messages)

                # Wrap output in an object with a 'content' attribute
                class ResultWrapper:
                    def __init__(self, content):
                        self.content = content

                return ResultWrapper(output_message.content)
            except Exception as e:
                attempt += 1
                logging.warning(
                    f"{Fore.YELLOW}Llama 3.1 invocation attempt {attempt}/{self.max_retries} failed: {e}"
                )
                time.sleep(delay)
                delay *= 2
        raise ValueError(
            f"Llama 3.1 invocation failed after {self.max_retries} attempts"
        )
