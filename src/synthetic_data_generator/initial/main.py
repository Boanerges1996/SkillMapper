import time
from agent import build_graph
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from colorama import init, Fore
from dotenv import load_dotenv

load_dotenv()

# Initialize Colorama
init(autoreset=True)

# Load Google API key from environment variable
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


if not os.environ["GOOGLE_API_KEY"]:
    raise ValueError("Google API key is not set in the environment variables.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    max_tokens=20000,
    temperature=0.7,
    top_p=0.9,
    frequency_penalty=0.5,
    timeout=30,
    max_retries=2,
)
print(f"{Fore.CYAN}🚀 Starting data generation process...")

start_time = time.perf_counter()  # Start timer

graph = build_graph(llm)
app = graph.compile()

initial_state = {
    "all_skills": [],
    "selected_skills": [],
    "course_data": [],
    "job_data": [],
    "cv_data": [],
    "all_data": [],
    "train_data": [],
    "val_data": [],
    "test_data": [],
}

final_state = app.invoke(initial_state)

end_time = time.perf_counter()  # End timer
elapsed = end_time - start_time

print(
    f"{Fore.GREEN}🎉 Data generation completed successfully in {elapsed:.2f} seconds!"
)
