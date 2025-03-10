import json
import logging
import math
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    SentencesDataset,
)

# Set up logging to track progress.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Starting SentenceTransformer finetuning script...")


# 1. Data Loading: Function to load a JSON file.
def load_json_file(filepath):
    logging.info(f"Loading JSON file from: {filepath}")
    with open(filepath, "r") as f:
        data = json.load(f)
    logging.info(f"Loaded {len(data)} records from {filepath}")
    return data


# Load the three JSON files.
course_data = load_json_file("course.json")
cv_data = load_json_file("cv.json")
job_data = load_json_file("job.json")

# Combine all records from the three files.
all_data = course_data + cv_data + job_data
logging.info(f"Total records combined: {len(all_data)}")

# 2. Create (preferredLabel, sentence) pairs from both explicit and implicit sentences.
pairs = []
for item in all_data:
    skill = item["preferredLabel"]
    sentences = item.get("sentences", {})
    for sentence in sentences.get("explicit", []):
        pairs.append((skill, sentence))
    for sentence in sentences.get("implicit", []):
        pairs.append((skill, sentence))
logging.info(f"Total pairs created: {len(pairs)}")

# 3. Convert each pair into an InputExample.
# An InputExample expects a list of texts; here we provide [skill, sentence].
train_examples = [InputExample(texts=[skill, sentence]) for skill, sentence in pairs]
logging.info(f"Created {len(train_examples)} training examples.")

# 4. Initialize a pre-trained SentenceTransformer model.
model_name = "all-mpnet-base-v2"
model = SentenceTransformer(model_name)
logging.info(f"Initialized SentenceTransformer model: {model_name}")

# 5. Create a SentencesDataset and DataLoader.
train_dataset = SentencesDataset(train_examples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
logging.info("Dataset and DataLoader created.")

# 6. Define the loss function.
# MultipleNegativesRankingLoss uses in-batch negatives, encouraging matching pairs to have higher similarity.
train_loss = losses.MultipleNegativesRankingLoss(model)
logging.info("Using MultipleNegativesRankingLoss for training.")

# 7. Set training parameters.
num_epochs = 3
warmup_steps = math.ceil(
    len(train_dataloader) * num_epochs * 0.1
)  # typically 10% of total training steps
logging.info(f"Training for {num_epochs} epochs with {warmup_steps} warmup steps.")

# 8. Fine-tune the model.
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path="finetuned_sentence_transformer",
)
logging.info("Training completed. Model saved to 'finetuned_sentence_transformer'.")
