import json
import os
import pandas as pd
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_meta import ChatMetaLLama
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
from sentence_transformers import SentenceTransformer, util
import re
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class SkillExtractorBERT:
    """BERT-based skill extractor using ModernBERT from HuggingFace"""
    
    def __init__(self, model_name="BAAI/bge-small-en-v1.5", threshold=0.75):
        """
        Initialize the BERT model for skill extraction
        Args:
            model_name: HuggingFace model name (default: BAAI/bge-small-en-v1.5, a modern BERT variant)
            threshold: Similarity threshold for skill matching
        """
        logger.info(f"Initializing BERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.threshold = threshold
        self.sentence_transformer = SentenceTransformer(model_name)
        
    def _extract_embedding(self, text):
        """Extract embeddings from text using the BERT model"""
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use CLS token embedding as text representation
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
    def extract_skills(self, text: str, all_skills: List[Dict[str, Any]], top_k=10) -> List[Dict[str, str]]:
        """
        Extract skills from text using semantic similarity
        
        Args:
            text: Input text to extract skills from
            all_skills: List of all possible skills with their embeddings
            top_k: Number of top skills to return
            
        Returns:
            List of extracted skills with their ESCO IDs
        """
        # Get text embedding
        text_embedding = self._extract_embedding(text)
        
        # Calculate similarity with all skills
        skill_scores = []
        for skill in all_skills:
            skill_emb = np.array(skill["embedding"])
            # Cosine similarity
            similarity = np.dot(text_embedding, skill_emb) / (np.linalg.norm(text_embedding) * np.linalg.norm(skill_emb))
            if similarity > self.threshold:
                skill_scores.append((skill, float(similarity)))
        
        # Sort by similarity score
        skill_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k skills above threshold
        return [
            {"skill_name": skill["skill_name"], "esco_id": skill["esco_id"]} 
            for skill, score in skill_scores[:top_k]
        ]

class SkillExtractorLLM:
    """Generative AI-based skill extractor"""
    
    def __init__(self, model_type="gemini", temperature=0, shot_setting="zero"):
        """
        Initialize the LLM for skill extraction
        Args:
            model_type: Type of LLM to use (gemini, gpt, claude, llama)
            temperature: Temperature for generation (lower = more deterministic)
            shot_setting: "zero" for zero-shot, "few" for few-shot
        """
        logger.info(f"Initializing LLM model: {model_type} ({shot_setting}-shot)")
        self.model_type = model_type
        self.shot_setting = shot_setting
        
        if model_type == "gemini":
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-pro",
                temperature=temperature,
            )
        elif model_type == "gpt":
            self.llm = ChatOpenAI(
                model="gpt-4o", 
                temperature=temperature,
            )
        elif model_type == "claude":
            self.llm = ChatAnthropic(
                model="claude-3-opus-20240229",
                temperature=temperature,
            )
        elif model_type == "llama":
            self.llm = ChatMetaLLama(
                model="llama-3.1-405b-instruct",
                temperature=temperature,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Embedding model for calculating similarity with ground truth
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
    def extract_skills(self, text: str, all_skills: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Extract skills from text using LLM
        
        Args:
            text: Input text to extract skills from
            all_skills: List of all possible skills
            
        Returns:
            List of extracted skills with their ESCO IDs
        """
        # Create prompt for skill extraction based on shot setting
        if self.shot_setting == "zero":
            prompt = self._create_zero_shot_prompt(text)
        else:  # few-shot
            prompt = self._create_few_shot_prompt(text)
        
        # Get response from LLM
        response = self.llm.invoke(prompt)
        response_text = response.content
        
        # Extract JSON from response
        json_match = re.search(r'\[.*\]', response_text.replace('\n', ' '), re.DOTALL)
        if json_match:
            try:
                skills_json = json.loads(json_match.group())
                # Map extracted skills to ESCO skills
                mapped_skills = self._map_to_esco(skills_json, all_skills)
                return mapped_skills
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from: {response_text}")
                return []
        else:
            logger.error(f"No JSON found in response: {response_text}")
            return []
    
    def _create_zero_shot_prompt(self, text: str) -> str:
        """Create a zero-shot prompt for skill extraction"""
        return f"""
You are a skill extraction system. Given the following text, extract all relevant skills mentioned.
Match each skill to the most appropriate skill from the ESCO taxonomy.

Text:
{text}

Output the skills in JSON format as a list of objects with 'skill_name'.
Only output valid, professional skills. Focus on specific, concrete skills rather than general concepts.
Don't make up skills that aren't in the text. Be precise and comprehensive.

Your response should be parseable JSON in this format:
[{{"skill_name": "skill1"}}, {{"skill_name": "skill2"}}]
"""
    
    def _create_few_shot_prompt(self, text: str) -> str:
        """Create a few-shot prompt for skill extraction with examples"""
        return f"""
You are a skill extraction system. Given the following text, extract all relevant skills mentioned.
Match each skill to the most appropriate skill from the ESCO taxonomy.

Here are a few examples:

Example 1:
Text: "The ideal candidate will have experience with Python programming, data analysis, and machine learning. They should be able to communicate effectively with stakeholders and work in an agile environment."
Skills: [
  {{"skill_name": "Python programming"}}, 
  {{"skill_name": "data analysis"}}, 
  {{"skill_name": "machine learning"}},
  {{"skill_name": "communicate effectively"}},
  {{"skill_name": "work in agile environment"}}
]

Example 2:
Text: "This course teaches advanced calculus concepts, linear algebra, and mathematical modeling. Students will develop problem-solving skills and learn to apply mathematics to real-world problems."
Skills: [
  {{"skill_name": "advanced calculus"}}, 
  {{"skill_name": "linear algebra"}}, 
  {{"skill_name": "mathematical modeling"}},
  {{"skill_name": "problem-solving"}},
  {{"skill_name": "apply mathematics"}}
]

Example 3:
Text: "Jane Smith has over 5 years of experience in digital marketing, with expertise in SEO, content creation, and social media management. She is proficient in using tools like Google Analytics, Ahrefs, and Adobe Creative Suite."
Skills: [
  {{"skill_name": "digital marketing"}}, 
  {{"skill_name": "SEO"}}, 
  {{"skill_name": "content creation"}},
  {{"skill_name": "social media management"}},
  {{"skill_name": "Google Analytics"}},
  {{"skill_name": "Ahrefs"}},
  {{"skill_name": "Adobe Creative Suite"}}
]

Now, extract skills from the following text:
{text}

Output the skills in JSON format as a list of objects with 'skill_name'.
Only output valid, professional skills. Focus on specific, concrete skills rather than general concepts.
Don't make up skills that aren't in the text. Be precise and comprehensive.

Your response should be parseable JSON in this format:
[{{"skill_name": "skill1"}}, {{"skill_name": "skill2"}}]
"""
    
    def _map_to_esco(self, extracted_skills: List[Dict[str, str]], all_skills: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Map extracted skills to ESCO skills based on semantic similarity"""
        mapped_skills = []
        
        # Get all skill names from ESCO
        all_skill_names = [skill["skill_name"] for skill in all_skills]
        all_skill_embeddings = self.embedding_model.encode(all_skill_names)
        
        for skill_obj in extracted_skills:
            skill_name = skill_obj.get("skill_name", "")
            if not skill_name:
                continue
                
            # Get embedding for extracted skill
            skill_embedding = self.embedding_model.encode(skill_name)
            
            # Calculate similarity with all ESCO skills
            similarities = util.cos_sim(skill_embedding, all_skill_embeddings)[0]
            best_match_idx = similarities.argmax().item()
            
            # If similarity is too low, skip
            if similarities[best_match_idx] < 0.5:
                continue
                
            # Get matched ESCO skill
            matched_skill = all_skills[best_match_idx]
            mapped_skills.append({
                "skill_name": matched_skill["skill_name"],
                "esco_id": matched_skill["esco_id"]
            })
            
        return mapped_skills

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def evaluate_extractor(
    extractor,
    data: List[Dict[str, Any]],
    all_skills: List[Dict[str, Any]],
    data_type: str,
    data_limit: int = 100
) -> Dict[str, float]:
    """
    Evaluate a skill extractor on a dataset
    
    Args:
        extractor: Skill extractor (BERT or LLM)
        data: Test data with ground truth skills
        all_skills: List of all possible skills
        data_type: Type of data (course, job, cv)
        data_limit: Maximum number of examples to evaluate
        
    Returns:
        Dictionary with evaluation metrics
    """
    all_predictions = []
    all_ground_truth = []
    extraction_times = []
    reciprocal_ranks = []
    
    # Use subset for evaluation
    subset = data[:min(data_limit, len(data))]
    
    for item in tqdm(subset, desc=f"Evaluating {data_type}"):
        # Get text to extract from
        if data_type == "course":
            text = f"{item['text']['courseTitle']}. {item['text']['description']}. {item['text']['learningOutcomes']}"
        elif data_type == "job":
            text = f"{item['text']['jobTitle']}. {item['text']['description']}. {item['text']['requirements']}"
        elif data_type == "cv":
            # Convert skills list to text for extraction
            skills_text = ", ".join(item['text']['skills'])
            text = f"{item['text']['personalInformation']['name']}. Skills: {skills_text}"
        
        # Ground truth skills
        ground_truth = [skill["skill_name"] for skill in item["skills"]]
        
        # Extract skills
        start_time = time.time()
        extracted_skills = extractor.extract_skills(text, all_skills)
        end_time = time.time()
        extraction_times.append(end_time - start_time)
        
        extracted_skill_names = [skill["skill_name"] for skill in extracted_skills]
        
        # Calculate Mean Reciprocal Rank (MRR)
        for gt_skill in ground_truth:
            if gt_skill in extracted_skill_names:
                rank = extracted_skill_names.index(gt_skill) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        
        # Convert to binary classification for metrics
        all_skill_names = set([skill["skill_name"] for skill in all_skills])
        y_true = [1 if skill in ground_truth else 0 for skill in all_skill_names]
        y_pred = [1 if skill in extracted_skill_names else 0 for skill in all_skill_names]
        
        all_ground_truth.extend(y_true)
        all_predictions.extend(y_pred)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_ground_truth, all_predictions, average='binary'
    )
    accuracy = accuracy_score(all_ground_truth, all_predictions)
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
    avg_time = np.mean(extraction_times)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "mrr": mrr,
        "avg_extraction_time": avg_time
    }

def run_comparison(models=["all"], data_limit=100):
    """
    Run the full comparison between BERT and generative AI models
    
    Args:
        models: List of models to include in comparison ("bert", "gpt", "gemini", "claude", "llama", "all")
        data_limit: Maximum number of examples to evaluate per data type
    """
    # Load ESCO skills
    logger.info("Loading ESCO skills...")
    all_skills = load_esco_skills("src/synthetic_data_generator/skills_en.csv")
    logger.info(f"Loaded {len(all_skills)} skills")
    
    # Load datasets
    logger.info("Loading datasets...")
    course_data = load_data("course_data.json")
    job_data = load_data("job_data.json")
    cv_data = load_data("cv_data.json")
    
    # Determine which models to include
    include_models = {}
    if "all" in models or "bert" in models:
        include_models["bert"] = True
    else:
        include_models["bert"] = False
        
    if "all" in models or "gpt" in models:
        include_models["gpt"] = True
    else:
        include_models["gpt"] = False
        
    if "all" in models or "gemini" in models:
        include_models["gemini"] = True
    else:
        include_models["gemini"] = False
        
    if "all" in models or "claude" in models:
        include_models["claude"] = True
    else:
        include_models["claude"] = False
        
    if "all" in models or "llama" in models:
        include_models["llama"] = True
    else:
        include_models["llama"] = False
    
    # Results dictionary
    results = {
        "model": [],
        "shot_setting": [],
        "data_type": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "accuracy": [],
        "mrr": [],
        "avg_extraction_time": []
    }
    
    # Evaluate BERT
    if include_models["bert"]:
        logger.info("Evaluating BERT model...")
        bert_extractor = SkillExtractorBERT()
        for data_type, data in [("course", course_data), ("job", job_data), ("cv", cv_data)]:
            metrics = evaluate_extractor(bert_extractor, data, all_skills, data_type, data_limit)
            results["model"].append("ModernBERT")
            results["shot_setting"].append("N/A")
            results["data_type"].append(data_type)
            for metric, value in metrics.items():
                results[metric].append(value)
            logger.info(f"BERT on {data_type}: {metrics}")
    
    # Evaluate LLMs with both zero-shot and few-shot settings
    llm_types = []
    if include_models["gemini"]:
        llm_types.append("gemini")
    if include_models["gpt"]:
        llm_types.append("gpt")
    if include_models["llama"]:
        llm_types.append("llama")
    if include_models["claude"]:
        llm_types.append("claude")
    
    for model_type in llm_types:
        for shot_setting in ["zero", "few"]:
            model_name_map = {"gemini": "Gemini", "gpt": "GPT-4o", "llama": "LLaMA-3.1", "claude": "Claude"}
            model_display_name = f"{model_name_map[model_type]} ({shot_setting}-shot)"
            logger.info(f"Evaluating {model_display_name}...")
            
            try:
                llm_extractor = SkillExtractorLLM(model_type=model_type, shot_setting=shot_setting)
                
                for data_type, data in [("course", course_data), ("job", job_data), ("cv", cv_data)]:
                    metrics = evaluate_extractor(llm_extractor, data, all_skills, data_type, data_limit)
                    results["model"].append(model_name_map[model_type])
                    results["shot_setting"].append(shot_setting)
                    results["data_type"].append(data_type)
                    for metric, value in metrics.items():
                        results[metric].append(value)
                    logger.info(f"{model_display_name} on {data_type}: {metrics}")
            except Exception as e:
                logger.error(f"Error evaluating {model_display_name}: {e}")
                logger.error("Skipping this model and continuing...")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv("skill_extraction_results.csv", index=False)
    logger.info("Results saved to skill_extraction_results.csv")
    
    # Create visualizations
    plot_results(results_df)
    
def plot_results(results_df):
    """Create visualizations of the results"""
    plt.figure(figsize=(16, 10))
    
    # Add model and shot setting together for plotting
    results_df['model_shot'] = results_df.apply(
        lambda row: f"{row['model']}" if row['shot_setting'] == 'N/A' 
        else f"{row['model']} ({row['shot_setting']}-shot)", axis=1
    )
    
    # F1 score by model and data type
    plt.subplot(2, 2, 1)
    sns.barplot(x='model_shot', y='f1', hue='data_type', data=results_df)
    plt.title('F1 Score by Model and Data Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Precision vs Recall
    plt.subplot(2, 2, 2)
    for model in results_df['model_shot'].unique():
        model_data = results_df[results_df['model_shot'] == model]
        plt.scatter(model_data['precision'], model_data['recall'], label=model)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.title('Precision vs Recall')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # MRR by model
    plt.subplot(2, 2, 3)
    sns.barplot(x='model_shot', y='mrr', data=results_df)
    plt.title('Mean Reciprocal Rank (MRR) by Model')
    plt.xticks(rotation=45, ha='right')
    
    # Extraction time by model
    plt.subplot(2, 2, 4)
    sns.barplot(x='model_shot', y='avg_extraction_time', data=results_df)
    plt.title('Average Extraction Time (s) by Model')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig("skill_extraction_comparison.png")
    logger.info("Visualizations saved to skill_extraction_comparison.png")
    
    # Create a separate plot comparing zero-shot vs few-shot
    plt.figure(figsize=(14, 8))
    
    # Filter to only LLM models
    llm_results = results_df[results_df['shot_setting'] != 'N/A']
    
    # Group by model, shot_setting, and aggregate by mean across data types
    shot_comparison = llm_results.groupby(['model', 'shot_setting'])[['precision', 'recall', 'f1', 'mrr']].mean().reset_index()
    
    # Melt for easier plotting
    shot_comparison_melted = pd.melt(
        shot_comparison, 
        id_vars=['model', 'shot_setting'], 
        value_vars=['precision', 'recall', 'f1', 'mrr'],
        var_name='Metric', 
        value_name='Value'
    )
    
    # Plot
    sns.barplot(x='model', y='Value', hue='shot_setting', col='Metric', data=shot_comparison_melted)
    plt.title('Zero-Shot vs Few-Shot Performance by Model and Metric')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("zero_vs_few_shot_comparison.png")
    logger.info("Zero-shot vs Few-shot comparison saved to zero_vs_few_shot_comparison.png")

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
    
    # Initialize embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Compute embeddings in batches to avoid memory issues
    batch_size = 128
    all_embeddings = []
    
    for i in range(0, len(skills), batch_size):
        batch = skills[i:i+batch_size]
        skill_texts = [skill["skill_name"] for skill in batch]
        embeddings = embedding_model.encode(skill_texts, convert_to_numpy=True)
        all_embeddings.extend(embeddings)
    
    # Add embeddings to skills
    for skill, emb in zip(skills, all_embeddings):
        skill["embedding"] = emb.tolist()
        
    return skills

if __name__ == "__main__":
    run_comparison()