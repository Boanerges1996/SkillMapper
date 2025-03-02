import pandas as pd
import numpy as np
import random
import json
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, validator

# Initialize the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def load_esco_skills(file_path: str) -> List[Dict[str, Any]]:
    """
    Load ESCO skills from a CSV file and precompute embeddings.
    """
    df = pd.read_csv(file_path)
    skills = (
        df[["preferredLabel", "conceptUri"]]
        .rename(columns={"preferredLabel": "skill_name", "conceptUri": "esco_id"})
        .to_dict(orient="records")
    )
    skill_texts = [skill["skill_name"] for skill in skills]
    embeddings = embedding_model.encode(skill_texts, convert_to_numpy=True)
    for skill, emb in zip(skills, embeddings):
        skill["embedding"] = emb.tolist()
    return skills


def get_relevant_skills(
    context: str, skills: List[Dict[str, Any]], top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Return the top_k skills most relevant to the context.
    """
    context_emb = embedding_model.encode(context, convert_to_numpy=True)

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    scored_skills = []
    for skill in skills:
        skill_emb = np.array(skill["embedding"])
        score = cosine_sim(context_emb, skill_emb)
        scored_skills.append((skill, score))
    scored_skills.sort(key=lambda x: x[1], reverse=True)
    return [skill for skill, _ in scored_skills[:top_k]]


def select_skills(all_skills: List[Dict[str, str]], num: int) -> List[Dict[str, str]]:
    if num > len(all_skills):
        raise ValueError("Requested number of skills exceeds available skills.")
    return random.sample(all_skills, num)


# ---------------------------
# Define Pydantic models and output parsers.


# Course Schema:
class CourseDescription(BaseModel):
    courseTitle: str
    courseCode: str
    credits: str
    description: str
    prerequisites: str
    learningOutcomes: List[str]


course_parser = PydanticOutputParser(pydantic_object=CourseDescription)


def generate_course_prompt() -> PromptTemplate:
    instructions = course_parser.get_format_instructions()
    escaped_instructions = instructions.replace("{", "{{").replace("}", "}}")
    prompt_text = (
        "Generate a JSON object for a course description that incorporates the following skills: {skills}.\n"
        + escaped_instructions
        + "\nDo not include any markdown formatting or extra text."
    )
    return PromptTemplate.from_template(prompt_text)


# Job Schema:
class JobDescription(BaseModel):
    jobTitle: str
    jobCode: str
    description: str
    requirements: str
    benefits: str


job_parser = PydanticOutputParser(pydantic_object=JobDescription)


def generate_job_prompt() -> PromptTemplate:
    instructions = job_parser.get_format_instructions()
    escaped_instructions = instructions.replace("{", "{{").replace("}", "}}")
    prompt_text = (
        "Generate a JSON object for a job description that incorporates the following skills: {skills}.\n"
        + escaped_instructions
        + "\nDo not include any markdown formatting or extra text."
    )
    return PromptTemplate.from_template(prompt_text)


# CV Schema:
class CVData(BaseModel):
    personalInformation: Dict[str, str]  # e.g., {"name": str, "contact": str}
    workExperience: List[Dict[str, str]]
    education: List[Dict[str, str]]
    skills: List[str]

    @validator("skills", pre=True)
    def flatten_skills(cls, v):
        if isinstance(v, list):
            flattened = []
            for item in v:
                if isinstance(item, list):
                    flattened.extend(item)
                else:
                    flattened.append(item)
            return flattened
        return v


cv_parser = PydanticOutputParser(pydantic_object=CVData)


def generate_cv_prompt() -> PromptTemplate:
    instructions = cv_parser.get_format_instructions()
    escaped_instructions = instructions.replace("{", "{{").replace("}", "}}")
    prompt_text = (
        "Generate a JSON object for a CV that incorporates the following skills: {skills}.\n"
        + escaped_instructions
        + "\nDo not include any markdown formatting or extra text."
    )
    return PromptTemplate.from_template(prompt_text)


# ---------------------------
# Helper for batching unique title generation.
def generate_unique_titles_stateful(
    generator_func, llm, total: int, batch_size: int = 100
) -> List[str]:
    unique_titles = set()
    attempts = 0
    current_batch_size = batch_size
    while len(unique_titles) < total and attempts < 10:
        remaining = total - len(unique_titles)
        try:
            new_titles = generator_func(
                llm,
                num_titles=min(current_batch_size, remaining),
                avoid=list(unique_titles),
            )
        except ValueError as e:
            print(f"Error in batch generation: {e}. Reducing batch size.")
            current_batch_size = max(10, current_batch_size // 2)
            continue
        before = len(unique_titles)
        unique_titles.update(new_titles)
        after = len(unique_titles)
        print(f"Generated {after} unique titles so far (attempt {attempts+1}).")
        attempts += 1
        if after == before:
            break
    return list(unique_titles)[:total]


# Updated title generators to accept an avoid list.
def generate_course_titles(
    llm, num_titles: int = 100, avoid: Optional[List[str]] = None
) -> List[str]:
    avoid_text = ""
    if avoid:
        avoid_text = " Avoid the following titles: " + ", ".join(avoid) + "."
    prompt_text = (
        f"Generate a JSON array containing {num_titles} realistic and diverse course titles from various academic fields."
        f"{avoid_text} Each course title should be a string. The output must be a valid JSON array of strings."
    )
    prompt = PromptTemplate.from_template(prompt_text)
    chain = prompt | llm
    result = chain.invoke({})
    text = result.content if hasattr(result, "content") else str(result)
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    if not text:
        raise ValueError("Empty output from course title generation.")
    try:
        course_titles = json.loads(text)
        if not isinstance(course_titles, list):
            raise ValueError("Output is not a list")
        return course_titles
    except Exception as e:
        raise ValueError(f"Could not parse course titles JSON: {e}. Raw output: {text}")


def generate_job_titles(
    llm, num_titles: int = 100, avoid: Optional[List[str]] = None
) -> List[str]:
    avoid_text = ""
    if avoid:
        avoid_text = " Avoid the following titles: " + ", ".join(avoid) + "."
    prompt_text = (
        f"Generate a JSON array containing {num_titles} realistic and diverse job titles from various industries."
        f"{avoid_text} Each job title should be a string. The output must be a valid JSON array of strings."
    )
    prompt = PromptTemplate.from_template(prompt_text)
    chain = prompt | llm
    result = chain.invoke({})
    text = result.content if hasattr(result, "content") else str(result)
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    if not text:
        raise ValueError("Empty output from job title generation.")
    try:
        job_titles = json.loads(text)
        if not isinstance(job_titles, list):
            raise ValueError("Output is not a list")
        return job_titles
    except Exception as e:
        raise ValueError(f"Could not parse job titles JSON: {e}. Raw output: {text}")


def generate_cv_names(
    llm, num_names: int = 100, avoid: Optional[List[str]] = None
) -> List[str]:
    avoid_text = ""
    if avoid:
        avoid_text = " Avoid the following names: " + ", ".join(avoid) + "."
    prompt_text = (
        f"Generate a JSON array containing {num_names} realistic candidate names for CVs."
        f"{avoid_text} Each candidate name should be a string. The output must be a valid JSON array of strings."
    )
    prompt = PromptTemplate.from_template(prompt_text)
    chain = prompt | llm
    result = chain.invoke({})
    text = result.content if hasattr(result, "content") else str(result)
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    if not text:
        raise ValueError("Empty output from CV name generation.")
    try:
        cv_names = json.loads(text)
        if not isinstance(cv_names, list):
            raise ValueError("Output is not a list")
        return cv_names
    except Exception as e:
        raise ValueError(f"Could not parse CV names JSON: {e}. Raw output: {text}")


# ---------------------------
# Structured generation helper functions.
def generate_course_example(
    all_skills: List[Dict[str, Any]], llm, course_title: str
) -> Dict[str, Any]:
    selected_skills = get_relevant_skills(course_title, all_skills, top_k=15)
    skill_names = [s["skill_name"] for s in selected_skills]
    inferred_context = infer_context("course", selected_skills, llm)
    extended_context = f"{inferred_context} Course Title: {course_title}."
    prompt = generate_course_prompt()
    chain = prompt | llm
    result = chain.invoke({"skills": ", ".join(skill_names)})
    try:
        parsed = course_parser.parse(result.content)
    except Exception as e:
        raise ValueError(f"Course output parsing failed for '{course_title}': {e}")
    parsed.courseTitle = course_title
    return {
        "text": parsed.dict(),
        "skills": [
            {"skill_name": s["skill_name"], "esco_id": s["esco_id"]}
            for s in selected_skills
        ],
    }


def generate_job_example(
    all_skills: List[Dict[str, Any]], llm, job_title: str
) -> Dict[str, Any]:
    selected_skills = get_relevant_skills(job_title, all_skills, top_k=15)
    skill_names = [s["skill_name"] for s in selected_skills]
    inferred_context = infer_context("job", selected_skills, llm)
    extended_context = f"{inferred_context} Job Title: {job_title}."
    prompt = generate_job_prompt()
    chain = prompt | llm
    result = chain.invoke({"skills": ", ".join(skill_names)})
    try:
        parsed = job_parser.parse(result.content)
    except Exception as e:
        raise ValueError(f"Job output parsing failed for '{job_title}': {e}")
    parsed.jobTitle = job_title
    return {
        "text": parsed.dict(),
        "skills": [
            {"skill_name": s["skill_name"], "esco_id": s["esco_id"]}
            for s in selected_skills
        ],
    }


def generate_cv_example(
    all_skills: List[Dict[str, Any]], llm, candidate_name: str
) -> Dict[str, Any]:
    selected_skills = get_relevant_skills(candidate_name, all_skills, top_k=15)
    skill_names = [s["skill_name"] for s in selected_skills]
    inferred_context = infer_context("cv", selected_skills, llm)
    extended_context = f"{inferred_context} Candidate Name: {candidate_name}."
    prompt = generate_cv_prompt()
    chain = prompt | llm
    result = chain.invoke({"skills": ", ".join(skill_names)})
    try:
        parsed = cv_parser.parse(result.content)
    except Exception as e:
        raise ValueError(f"CV output parsing failed for '{candidate_name}': {e}")
    if "personalInformation" in parsed.dict():
        parsed.personalInformation["name"] = candidate_name
    else:
        parsed.personalInformation = {"name": candidate_name, "contact": ""}
    return {
        "text": parsed.dict(),
        "skills": [
            {"skill_name": s["skill_name"], "esco_id": s["esco_id"]}
            for s in selected_skills
        ],
    }


# ---------------------------
# infer_context function added here.
def infer_context(data_type: str, skills: List[Dict[str, Any]], llm) -> str:
    """
    Use the LLM to generate a brief context description based on the list of skills.
    """
    if data_type == "course":
        prompt_text = (
            "Based on the following skills: {skills}, generate a brief context description for a course. "
            "Describe the subject area and style of the course in one or two sentences."
        )
    elif data_type == "cv":
        prompt_text = (
            "Based on the following skills: {skills}, generate a brief context description for a CV. "
            "Describe the professional background and focus areas in one or two sentences."
        )
    else:
        prompt_text = "Based on the following skills: {skills}, generate a brief context description in one or two sentences."
    prompt = PromptTemplate.from_template(prompt_text)
    chain = prompt | llm
    skills_str = ", ".join([s["skill_name"] for s in skills])
    result = chain.invoke({"skills": skills_str})
    context = result.content if hasattr(result, "content") else str(result)
    return context.strip()
