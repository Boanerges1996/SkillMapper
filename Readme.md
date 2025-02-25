# Skills Extraction and Mapping Framework

## Overview

This framework is designed for extracting and mapping skills to the [ESCO](https://ec.europa.eu/esco/portal/home) (European Skills, Competences, Qualifications and Occupations) taxonomy. It also includes tools for generating synthetic data and performing comparative analyses between different language models, such as [BERT](https://github.com/google-research/bert) and various Generative AI models (e.g., [GPT-4](https://openai.com/research/gpt-4), [LLaMA 3.1](https://ai.facebook.com/blog/large-language-models-llama/), [DeepSeek R1](https://www.deepseek.ai/)).

## Features

- **Skills Extraction**: Extract skills from various text sources and map them to ESCO.
- **Synthetic Data Generation**: Generate synthetic job descriptions, course descriptions, and CV data using AI agents.
- **Comparative Analysis**: Perform comparative analyses between different language models, including BERT and Generative AI models like GPT-4, LLaMA 3.1, and DeepSeek R1.
- **AI Agent**: Utilize [LangGraph](langchain-ai.github.io) to build an AI agent for generating synthetic data.

## Codebase Structure

### `/src`

- **`synthetic_data_generator/tools.py`**: Contains the main tools and functions for generating synthetic data and performing skill extraction and mapping.

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

   - Copy the `.env.example` file to `.env` and fill in the required values.
   - Example:
     ```sh
     cp .env.example .env
     ```

4. **Run the tools**:
   - To generate synthetic data, use the functions provided in `tools.py`.
   - To perform skill extraction and mapping, load the ESCO skills and use the relevant functions.

## AI Agent with LangGraph

The framework leverages [LangGraph](https://langgraph.ai/) to build an AI agent for generating synthetic data. LangGraph provides a powerful and flexible way to create and manage AI workflows, making it easier to generate high-quality synthetic data for various use cases.

## Comparative Analysis

The framework includes tools for performing comparative analyses between different language models. You can use the provided functions to generate synthetic data and compare the performance of models like [BERT](https://github.com/google-research/bert), [GPT-4](https://openai.com/research/gpt-4), [LLaMA 3.1](https://ai.facebook.com/blog/large-language-models-llama/), and [DeepSeek R1](https://www.deepseek.ai/).

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the [MIT License](./LICENSE).
