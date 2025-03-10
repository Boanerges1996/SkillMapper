import os
import json
import math
import logging
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    SentencesDataset,
)

# Limit CPU usage to 10 threads
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
import torch

torch.set_num_threads(10)

# Set up logging configuration to track progress.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Starting SentenceTransformer training script with limited CPU threads...")


# Function to load a JSON file
def load_json_file(filepath):
    logging.info(f"Loading JSON file from: {filepath}")
    with open(filepath, "r") as f:
        data = json.load(f)
    logging.info(f"Loaded {len(data)} records from {filepath}")
    return data


# Load the three JSON files
course_data = load_json_file("course.json")
cv_data = load_json_file("cv.json")
job_data = load_json_file("job.json")

# Combine all records from the three files
all_data = course_data + cv_data + job_data
logging.info(f"Total records combined: {len(all_data)}")

# Create (preferredLabel, sentence) pairs from both explicit and implicit sentences.
pairs = []
for item in all_data:
    skill = item["preferredLabel"]
    sentences = item.get("sentences", {})
    for sentence in sentences.get("explicit", []):
        pairs.append((skill, sentence))
    for sentence in sentences.get("implicit", []):
        pairs.append((skill, sentence))
logging.info(f"Total pairs created: {len(pairs)}")

# Convert each (skill, sentence) pair into an InputExample object required by SentenceTransformer.
train_examples = []
for pair in pairs:
    # Each InputExample takes a list of texts; here we provide two texts: [skill, sentence]
    train_examples.append(InputExample(texts=[pair[0], pair[1]]))
logging.info(f"Created {len(train_examples)} training examples.")

# Initialize a SentenceTransformer model. We use 'all-mpnet-base-v2' as an example.
model_name = "all-mpnet-base-v2"
model = SentenceTransformer(model_name)
logging.info(f"Initialized SentenceTransformer model: {model_name}")

# Create a SentencesDataset using the training examples and the model.
train_dataset = SentencesDataset(train_examples, model)
# Limit DataLoader workers to 10 to use about 10 CPUs
train_dataloader = DataLoader(
    train_dataset, shuffle=True, batch_size=32, num_workers=10
)

# Define the loss function.
# MultipleNegativesRankingLoss automatically uses in-batch negatives to encourage the model
# to pull together embeddings of matching pairs while pushing apart non-matching ones.
train_loss = losses.MultipleNegativesRankingLoss(model)
logging.info("Using MultipleNegativesRankingLoss for training.")

# Set training parameters
num_epochs = 3
warmup_steps = math.ceil(
    len(train_dataloader) * num_epochs * 0.1
)  # typically 10% of total training steps
logging.info(f"Training for {num_epochs} epochs with {warmup_steps} warmup steps.")

# Fine-tune the SentenceTransformer model using its built-in fit function.
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path="finetuned_sentence_transformer",
)

logging.info("Training completed. Model saved to 'finetuned_sentence_transformer'.")
