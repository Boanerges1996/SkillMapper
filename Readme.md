# Skills Extraction and Mapping Framework

## Overview

This framework is designed for extracting and mapping skills to the [ESCO](https://ec.europa.eu/esco/portal/home) (European Skills, Competences, Qualifications and Occupations) taxonomy. It also includes tools for generating synthetic data and performing comparative analyses between different language models, such as [BERT](https://github.com/google-research/bert) and various Generative AI models (e.g., [GPT-4o](https://openai.com/gpt-4o), [LLaMA 3.1](https://ai.facebook.com/blog/llama-3/), [Gemini](https://ai.google.dev/gemini), and [Claude](https://www.anthropic.com/claude)).

## Features

- **Skills Extraction**: Extract skills from various text sources and map them to ESCO.
- **Synthetic Data Generation**: Generate synthetic job descriptions, course descriptions, and CV data using AI agents.
- **Comparative Analysis**: Perform comparative analyses between different language models, including BERT and Generative AI models.
- **AI Agent**: Utilize [LangGraph](https://langchain-ai.github.io/langgraph/) to build an AI agent for generating synthetic data.

## Codebase Structure

### `/src`
- **`synthetic_data_generator/`**: Contains the tools for generating synthetic data.
  - **`tools.py`**: Core functions for synthetic data generation and skill extraction.
  - **`agent.py`**: LangGraph agent implementation for data generation.
  - **`main.py`**: Entry point for running the synthetic data generator.
- **`skill_extraction_comparison.py`**: Script to run comparative analysis between models.
- **`comparison_results_analysis.py`**: Analysis tools for visualizing and interpreting results.
- **`run_comparative_study.py`**: Main script to run the full comparative study pipeline.

## Key Components

- **ESCO Skills Loading**: Load ESCO skills from a CSV file and precompute embeddings for efficient skill matching.
- **Skill Relevance**: Determine the most relevant skills for a given context using cosine similarity.
- **Synthetic Data Generation**: Generate synthetic job descriptions, course descriptions, and CV data using predefined schemas and prompt templates.
- **Title Generation**: Generate unique and diverse titles for courses, jobs, and CVs.
- **Context Inference**: Use the LLM to generate brief context descriptions based on a list of skills.

## Getting Started

1. **Clone the repository**:

   ```sh
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install dependencies**:

   ```sh
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:

   - Create a `.env` file and add your API keys:
     ```
     OPENAI_API_KEY=your_openai_key
     GOOGLE_API_KEY=your_google_key
     ANTHROPIC_API_KEY=your_anthropic_key
     ```

4. **Generate synthetic data**:
   ```sh
   python src/synthetic_data_generator/main.py
   ```

## Comparative Study: BERT vs. Generative AI for Skill Extraction

### Overview

This project includes a comprehensive comparison between BERT-based models (Answer.AI's ModernBERT-base) and several Generative AI models (GPT-4o, LLaMA 3.1, Gemini, Claude) for the task of skill extraction from three types of documents:

1. Course descriptions
2. Job postings
3. CVs (resumes)

The comparison evaluates models on standard metrics including precision, recall, F1 score, accuracy, and Mean Reciprocal Rank (MRR).

### Zero-Shot vs. Few-Shot Evaluation

The generative AI models are evaluated in two prompting settings:

1. **Zero-shot**: Models extract skills without any examples in the prompt
2. **Few-shot**: Models are provided with 3 examples of skill extraction in different contexts

This allows us to analyze:
- How examples affect performance across different models
- Whether some models benefit more from examples than others
- The trade-offs between precision and recall in zero-shot vs. few-shot settings

### Running the Comparative Study

```bash
# Run the full comparison with all models
python src/run_comparative_study.py

# Run with specific models only
python src/run_comparative_study.py --models bert gpt gemini

# Run only the analysis on existing results
python src/run_comparative_study.py --skip-comparison
```

### Methodology

The project implements two distinct approaches to skill extraction:

1. **Embedding-based approach (BERT)**: Uses semantic similarity between document text and skill descriptions to identify relevant skills
2. **Generative approach (LLMs)**: Prompts the language model to extract skills from the document text, then maps these to ESCO skills

Each model is evaluated on the same test set, and performance metrics are calculated using sklearn.

### Results

The results of the study will be saved to:

- `skill_extraction_results.csv`: Raw metrics for each model and document type
- `analysis_output/`: Directory containing visualizations and analysis
  - Performance charts
  - Statistical significance tests
  - Zero-shot vs. few-shot comparisons
  - Research findings summary (in `research_findings.md`)

## AI Agent with LangGraph

The framework leverages [LangGraph](https://langchain-ai.github.io/langgraph/) to build an AI agent for generating synthetic data. LangGraph provides a powerful and flexible way to create and manage AI workflows, making it easier to generate high-quality synthetic data for various use cases.

The agent pipeline:
1. Loads ESCO skills from a CSV file
2. Generates unique titles for courses, jobs, and CVs
3. Creates detailed synthetic documents for each title
4. Maps skills to standardized ESCO IDs
5. Saves everything to JSON files

## Extending the Framework

The code is designed to be easily extended with:

- Additional models
- Different document types
- Custom evaluation metrics
- Alternative skill taxonomies
- Different few-shot prompting strategies

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the [MIT License](./LICENSE).