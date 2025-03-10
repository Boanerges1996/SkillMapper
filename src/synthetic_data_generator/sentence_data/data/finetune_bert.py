import json
import os
import logging
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Starting the script...")


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

# Combine all entries into one list
all_data = course_data + cv_data + job_data
logging.info(f"Total records combined: {len(all_data)}")

# Create (preferredLabel, sentence) pairs from both explicit and implicit sentences.
pairs = []
for item in all_data:
    skill = item["preferredLabel"]
    sentences = item.get("sentences", {})
    # Use explicit sentences if available.
    for sentence in sentences.get("explicit", []):
        pairs.append((skill, sentence))
    # And implicit sentences.
    for sentence in sentences.get("implicit", []):
        pairs.append((skill, sentence))

logging.info(f"Total pairs created: {len(pairs)}")
# Optionally, inspect a few pairs:
logging.info("Sample pairs:")
for pair in pairs[:3]:
    logging.info(pair)


class SkillSentenceDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
        # Initialize the tokenizer here for potential use later if needed.
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        logging.info("SkillSentenceDataset initialized.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        skill, sentence = self.pairs[idx]
        return {"skill": skill, "sentence": sentence}


# A collate function to group skills and sentences
def collate_fn(batch):
    skills = [item["skill"] for item in batch]
    sentences = [item["sentence"] for item in batch]
    return skills, sentences


# Create dataset and DataLoader
logging.info("Creating dataset and DataLoader...")
dataset = SkillSentenceDataset(pairs)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
logging.info("Dataset and DataLoader created.")


class BiEncoderModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(BiEncoderModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        logging.info(f"BiEncoderModel initialized with {model_name}.")

    def encode(self, texts, max_length=128):
        # Tokenize the input texts
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        # Move tensors to the same device as the model
        inputs = {
            k: v.to(next(self.bert.parameters()).device) for k, v in inputs.items()
        }
        outputs = self.bert(**inputs)
        # Use the [CLS] token representation as the sentence embedding
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings

    def forward(self, skill_texts, sentence_texts):
        skill_embeddings = self.encode(skill_texts)
        sentence_embeddings = self.encode(sentence_texts)
        return skill_embeddings, sentence_embeddings


def contrastive_loss(skill_embeddings, sentence_embeddings, temperature=0.05):
    # Normalize the embeddings to unit vectors
    skill_norm = F.normalize(skill_embeddings, p=2, dim=1)
    sentence_norm = F.normalize(sentence_embeddings, p=2, dim=1)

    # Compute the cosine similarity matrix (scaled by temperature)
    sim_matrix = torch.matmul(skill_norm, sentence_norm.T) / temperature

    # Create target labels - the diagonal elements are the positive pairs
    batch_size = skill_embeddings.size(0)
    labels = torch.arange(batch_size).to(skill_embeddings.device)

    # Compute cross-entropy loss across the similarity matrix rows
    loss = F.cross_entropy(sim_matrix, labels)
    return loss


# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Initialize our bi-encoder model and move it to the device
logging.info("Initializing BiEncoderModel...")
model = BiEncoderModel().to(device)

# Define an optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
logging.info("Optimizer initialized.")

# Training loop
num_epochs = 3
logging.info("Starting training...")
model.train()

for epoch in range(num_epochs):
    total_loss = 0
    logging.info(f"Epoch {epoch+1}/{num_epochs} started.")
    for batch_idx, (skills, sentences) in enumerate(dataloader):
        optimizer.zero_grad()
        # Encode both the skill texts and the corresponding sentences
        skill_emb, sent_emb = model(skills, sentences)
        loss = contrastive_loss(skill_emb, sent_emb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 10 == 0:
            logging.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    logging.info(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

logging.info("Training completed.")

# Save the model's state dictionary
logging.info("Saving fine-tuned model state dictionary...")
torch.save(model.state_dict(), "finetuned_bert.pth")
logging.info("Model state dictionary saved as 'finetuned_bert.pth'.")

# Save the tokenizer for future use
logging.info("Saving tokenizer...")
model.tokenizer.save_pretrained("finetuned_tokenizer")
logging.info("Tokenizer saved in 'finetuned_tokenizer'.")
logging.info("Script completed successfully.")
