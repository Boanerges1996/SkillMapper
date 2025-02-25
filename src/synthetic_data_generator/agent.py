import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from colorama import init, Fore
from langgraph.graph import StateGraph
from typing import TypedDict, List, Dict, Any
import json
from tools import (
    load_esco_skills,
    generate_course_titles,
    generate_job_titles,
    generate_cv_names,
    generate_course_example,
    generate_job_example,
    generate_cv_example,
    generate_unique_titles_stateful,
)

init(autoreset=True)
logging.basicConfig(level=logging.INFO, format="%(message)s")

MAX_RETRIES = 3


class State(TypedDict):
    all_skills: List[Dict[str, str]]
    selected_skills: List[Dict[str, str]]
    course_data: List[Dict[str, Any]]
    job_data: List[Dict[str, Any]]
    cv_data: List[Dict[str, Any]]
    all_data: List[Dict[str, Any]]
    train_data: List[Dict[str, Any]]
    val_data: List[Dict[str, Any]]
    test_data: List[Dict[str, Any]]
    num_course_titles: int
    num_job_titles: int
    num_cv_names: int


def generate_single_course(
    course_title: str, all_skills: List[Dict[str, Any]], llm: Any
) -> Dict[str, Any]:
    logging.info(f"{Fore.MAGENTA}🚀 Generating course for title: {course_title}")
    result = generate_course_example(all_skills, llm, course_title)
    logging.info(f"{Fore.GREEN}✅ Finished generating course: {course_title}")
    return result


def generate_single_job(
    job_title: str, all_skills: List[Dict[str, Any]], llm: Any
) -> Dict[str, Any]:
    logging.info(f"{Fore.MAGENTA}🚀 Generating job for title: {job_title}")
    result = generate_job_example(all_skills, llm, job_title)
    logging.info(f"{Fore.GREEN}✅ Finished generating job: {job_title}")
    return result


def generate_single_cv(
    cv_name: str, all_skills: List[Dict[str, Any]], llm: Any
) -> Dict[str, Any]:
    logging.info(f"{Fore.MAGENTA}🚀 Generating CV for candidate: {cv_name}")
    result = generate_cv_example(all_skills, llm, cv_name)
    logging.info(f"{Fore.GREEN}✅ Finished generating CV for candidate: {cv_name}")
    return result


def build_graph(llm):
    graph = StateGraph(state_schema=State)

    def load_esco_node(state):
        logging.info(f"{Fore.CYAN}🔍 Starting to load ESCO skills...")
        all_skills = load_esco_skills("skills_en.csv")
        logging.info(f"{Fore.GREEN}✅ Loaded {len(all_skills)} skills from ESCO.")
        return {"all_skills": all_skills}

    def select_skills_node(state):
        logging.info(
            f"{Fore.CYAN}📝 Ready to use full ESCO skills dataset for dynamic selection."
        )
        return {"selected_skills": state["all_skills"]}

    def generate_course_data_node(state):
        start_time = time.perf_counter()
        logging.info(
            f"{Fore.MAGENTA}🎓 Generating course data using Gemini model: {llm.model}"
        )
        total_course_titles = state.get("num_course_titles", 2500)
        # Use our stateful generator to obtain unique course titles in batches.
        course_titles = generate_unique_titles_stateful(
            generate_course_titles, llm, total=total_course_titles, batch_size=100
        )
        logging.info(
            f"{Fore.YELLOW}📚 Generated {len(course_titles)} unique course titles:"
        )
        logging.info(f"{Fore.YELLOW}{course_titles}")
        course_data = []
        total_courses = len(course_titles)
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(
                    generate_single_course, title, state["all_skills"], llm
                ): title
                for title in course_titles
            }
            for idx, future in enumerate(as_completed(futures), start=1):
                original_title = futures[future]
                try:
                    result = future.result()
                    course_data.append(result)
                    logging.info(
                        f"{Fore.CYAN}({idx}/{total_courses}) Course generated: {result['text'].get('courseTitle', 'N/A')}"
                    )
                except Exception as e:
                    logging.error(
                        f"{Fore.RED}Error generating course for '{original_title}': {e}"
                    )
        elapsed = time.perf_counter() - start_time
        logging.info(
            f"{Fore.GREEN}✅ Finished generating course data in {elapsed:.2f} seconds."
        )
        return {"course_data": course_data}

    def generate_job_data_node(state):
        start_time = time.perf_counter()
        logging.info(
            f"{Fore.MAGENTA}💼 Generating job data using Gemini model: {llm.model}"
        )
        total_job_titles = state.get("num_job_titles", 2000)
        job_titles = generate_unique_titles_stateful(
            generate_job_titles, llm, total=total_job_titles, batch_size=100
        )
        logging.info(f"{Fore.YELLOW}📋 Generated {len(job_titles)} unique job titles:")
        logging.info(f"{Fore.YELLOW}{job_titles}")
        job_data = []
        total_jobs = len(job_titles)
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(
                    generate_single_job, title, state["all_skills"], llm
                ): title
                for title in job_titles
            }
            for idx, future in enumerate(as_completed(futures), start=1):
                original_title = futures[future]
                try:
                    result = future.result()
                    job_data.append(result)
                    logging.info(
                        f"{Fore.CYAN}({idx}/{total_jobs}) Job generated: {result['text'].get('jobTitle', 'N/A')}"
                    )
                except Exception as e:
                    logging.error(
                        f"{Fore.RED}Error generating job for '{original_title}': {e}"
                    )
        elapsed = time.perf_counter() - start_time
        logging.info(
            f"{Fore.GREEN}✅ Finished generating job data in {elapsed:.2f} seconds."
        )
        return {"job_data": job_data}

    def generate_cv_data_node(state):
        start_time = time.perf_counter()
        logging.info(
            f"{Fore.MAGENTA}📝 Generating CV data using Gemini model: {llm.model}"
        )
        total_cv_names = state.get("num_cv_names", 1400)
        cv_names = generate_unique_titles_stateful(
            lambda llm, num_titles, avoid: generate_cv_names(
                llm, num_names=num_titles, avoid=avoid
            ),
            llm,
            total=total_cv_names,
            batch_size=100,
        )

        logging.info(
            f"{Fore.YELLOW}👤 Generated {len(cv_names)} unique candidate names for CVs:"
        )
        logging.info(f"{Fore.YELLOW}{cv_names}")
        cv_data = []
        total_cvs = len(cv_names)
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(
                    generate_single_cv, name, state["all_skills"], llm
                ): name
                for name in cv_names
            }
            for idx, future in enumerate(as_completed(futures), start=1):
                original_name = futures[future]
                try:
                    result = future.result()
                    cv_data.append(result)
                    logging.info(
                        f"{Fore.CYAN}({idx}/{total_cvs}) CV generated for candidate: {result['text'].get('personalInformation', {}).get('name', 'N/A')}"
                    )
                except Exception as e:
                    logging.error(
                        f"{Fore.RED}Error generating CV for '{original_name}': {e}"
                    )
        elapsed = time.perf_counter() - start_time
        logging.info(
            f"{Fore.GREEN}✅ Finished generating CV data in {elapsed:.2f} seconds."
        )
        return {"cv_data": cv_data}

    def save_data_node(state):
        logging.info(f"{Fore.CYAN}💾 Saving generated data to JSON files...")
        with open("./data/course_data.json", "w") as f:
            json.dump(state["course_data"], f)
        with open("./data/job_data.json", "w") as f:
            json.dump(state["job_data"], f)
        with open("./data/cv_data.json", "w") as f:
            json.dump(state["cv_data"], f)
        logging.info(f"{Fore.GREEN}✅ Data saved successfully.")
        return {}

    graph.add_node("load_esco", load_esco_node)
    graph.add_node("select_skills", select_skills_node)
    graph.add_node("generate_course", generate_course_data_node)
    graph.add_node("generate_job", generate_job_data_node)
    graph.add_node("generate_cv", generate_cv_data_node)
    graph.add_node("save_data", save_data_node)

    graph.set_entry_point("load_esco")
    graph.add_edge("load_esco", "select_skills")
    graph.add_edge("select_skills", "generate_course")
    graph.add_edge("generate_course", "generate_job")
    graph.add_edge("generate_job", "generate_cv")
    graph.add_edge("generate_cv", "save_data")
    graph.set_finish_point("save_data")
    return graph
