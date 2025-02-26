# Comparative Study: BERT vs. Generative AI for Skill Extraction

## Overview

This project provides a comprehensive comparison between BERT-based models (ModernBERT) and several Generative AI models (GPT-4o, LLaMA 3.1, Gemini, Claude) for the task of skill extraction from three types of documents:

1. Course descriptions
2. Job postings
3. CVs (resumes)

The comparison evaluates models on standard metrics including precision, recall, F1 score, accuracy, and Mean Reciprocal Rank (MRR).

## Research Questions

1. How do BERT-based models compare to generative AI models for skill extraction?
2. Which model performs best for different document types?
3. What are the trade-offs between precision and recall across models?
4. How does extraction speed compare across models?
5. What are the practical implications for building real-world skill extraction systems?

## Dataset

The study uses a synthetic dataset generated using the SkillMapper framework, which includes:

- Course descriptions with learning outcomes
- Job descriptions with requirements
- CV data with candidate skills

Each document is annotated with ground truth skills from the ESCO (European Skills, Competences, Qualifications and Occupations) taxonomy.

## Models Compared

1. **ModernBERT**: A modern BERT variant using BAAI/bge-small-en-v1.5
2. **GPT-4o**: OpenAI's latest GPT model
3. **LLaMA 3.1**: Meta's open-source large language model
4. **Gemini 2.0 Pro**: Google's generative AI model
5. **Claude 3 Opus**: Anthropic's latest generative AI model

## Methodology

The project implements two distinct approaches to skill extraction:

1. **Embedding-based approach (BERT)**: Uses semantic similarity between document text and skill descriptions to identify relevant skills
2. **Generative approach (LLMs)**: Prompts the language model to extract skills from the document text, then maps these to ESCO skills

Each model is evaluated on the same test set, and performance metrics are calculated using sklearn.

## Running the Study

### Prerequisites

```
pip install -r requirements.txt
```

### Running the Comparison

```bash
# Step 1: Run the skill extraction comparison
python src/skill_extraction_comparison.py

# Step 2: Analyze results and generate visualizations
python src/comparison_results_analysis.py
```

## Results

The results of the study will be saved to:

- `skill_extraction_results.csv`: Raw metrics for each model and document type
- `analysis_output/`: Directory containing visualizations and analysis
  - Performance charts
  - Statistical significance tests
  - Research findings summary

## API Keys

To run the full comparison, you'll need API keys for the following services:

- OpenAI (for GPT-4o)
- Google AI (for Gemini)
- Anthropic (for Claude)
- Meta (for Llama, if using hosted endpoints)

Add these to your environment variables or a `.env` file.

## Extending the Study

The code is designed to be easily extended with:

- Additional models
- Different document types
- Custom evaluation metrics
- Alternative skill taxonomies

## Citation

If you use this framework in your research, please cite:

```
@misc{skillmapper2025,
  author = {Samson Kwaku Nkrumah},
  title = {SkillMapper: A Comparative Study of BERT vs. Generative AI for Skill Extraction},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/SkillMapper}}
}
```
