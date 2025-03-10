import json
import logging
import torch
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, BertModel

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Starting model loading and testing script...")


# 1. Define the BiEncoderModel class (same as used during training)
class BiEncoderModel(torch.nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(BiEncoderModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        logging.info(f"BiEncoderModel initialized with {model_name}.")

    def encode(self, texts, max_length=128):
        # Tokenize the texts
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        # Move tensors to the same device as the model
        device = next(self.bert.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.bert(**inputs)
        # Use the [CLS] token as the sentence embedding
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings

    def forward(self, skill_texts, sentence_texts):
        skill_embeddings = self.encode(skill_texts)
        sentence_embeddings = self.encode(sentence_texts)
        return skill_embeddings, sentence_embeddings


# 2. Set device and load the model state dictionary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Instantiate the model and move to the device
model = BiEncoderModel().to(device)

# Load the saved state dictionary
model_path = "finetuned_bert.pth"
logging.info(f"Loading model state dictionary from {model_path}...")
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
logging.info("Model state dictionary loaded successfully.")

# Also load the tokenizer (saved during training)
tokenizer = BertTokenizer.from_pretrained("finetuned_tokenizer")
model.tokenizer = tokenizer  # update model tokenizer if needed
logging.info("Tokenizer loaded successfully from 'finetuned_tokenizer'.")


# 3. Load your data to obtain the list of unique skills
# Assuming you have the combined all_data from your training script.
def load_json_file(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


course_data = load_json_file("course.json")
cv_data = load_json_file("cv.json")
job_data = load_json_file("job.json")
all_data = course_data + cv_data + job_data

unique_skills = list({item["preferredLabel"] for item in all_data})
logging.info(f"Number of unique skills: {len(unique_skills)}")

# 4. Precompute embeddings for each unique skill
skill_embeddings = {}
model.eval()
with torch.no_grad():
    for skill in unique_skills:
        emb = model.encode([skill])
        # Store on CPU to free up GPU memory for inference
        skill_embeddings[skill] = emb.cpu()
logging.info("Precomputed embeddings for all unique skills.")


# 5. Define a function to predict top matching skills for a given input text
def predict_skills(course_text, top_k=10):
    model.eval()
    with torch.no_grad():
        # Compute embedding for the input course description
        course_emb = model.encode([course_text])
        course_emb = F.normalize(course_emb, p=2, dim=1)
        course_emb = course_emb.cpu()  # Move to CPU for compatibility

    sims = {}
    # Compare with each precomputed skill embedding (on CPU)
    for skill, emb in skill_embeddings.items():
        emb_norm = F.normalize(emb, p=2, dim=1)
        # Compute cosine similarity; both tensors are now on CPU
        sim = torch.matmul(course_emb, emb_norm.T).item()
        sims[skill] = sim

    # Sort skills by similarity in descending order
    sorted_skills = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    return sorted_skills[:top_k]


# 6. Test the model with a sample course description
course_example = """
Professional Summary
Results-driven software developer with expertise in full-stack development and creating robust, scalable applications.
Proficient in multiple programming languages, frameworks, and cloud technologies. Strong problem-solving skills and
dedicated to staying updated with emerging technologies. Experienced in delivering high-quality solutions within deadlines
and budgets. Note, I am currently exploring the machine learning field, CNN, RNN, Transformers, LLM’s, etc.
Education
Lovely Professional University Punjab, India
B. Tech Computer Science and Engineering August 2019 – June 2023
Munster Technological University Kerry, Ireland
Msc. Computer Science (By Research) July 2024 – June 2026
Experience
Software Engineer April 2023 – Present
Webrtc Ventures/Agilityfeat Charlottesville, VA, USA
• Developed Custom Live Video Recording Solution: Designed and implemented a live video recording system using
the Amazon Chime SDK, ensuring instant video availability through advanced stream processing and storage
techniques.
• Automated QA Testing: Automated QA processes using Loadero, significantly enhancing productivity by enabling
the efficient generation of multiple user profiles for testing.
• Integrated AWS Services: Improved an internal project by integrating AWS technologies, including Amazon Chime
SDK, Amplify, and Cognito, while adding new features and resolving critical system bugs.
• Built High-Throughput Custom Protocol: Developed a custom high-throughput protocol for a geo-satellite in
Canada, enabling communication services such as chat, email, SMS, and voice in remote areas.
• Implemented Real-Time Transcription and Analytics: Delivered real-time transcription and analytics solutions in
surgical settings by integrating Symbl AI, accompanied by a dashboard for operational monitoring
• Optimized Satellite Communication System Design: Designed and architected a high-throughput communication
system for a geo-satellite in Canada, implementing advanced congestion control mechanisms that significantly
increased data throughput, enhanced system reliability, and ensured seamless communication in remote areas
• Technologies Used: Node.js, Express.js, Nest.js, React.js, Flask, TypeScript, Next.js, MongoDB, PostgreSQL,
DynamoDB, AWS (Chime SDK, Amplify, Cognito, ECS, Lambda), Symbl AI, Docker, Redis, GraphQL, Janus,
SIP, Satellite Communication, etc.
Software Engineer/Architect (Full-time) September 2021 – March 2023
Reprezentme Ltd Abuja, Nigeria
• Led the design and architecture of the entire Reprezentme system.
• Integrated Work Methods: Successfully integrated work methods to align organizational goals with the corporate
mission, leading to significant business growth.
• End-to-End Encryption: Developed and deployed an end-to-end encryption solution for group messaging,
significantly improving the confidentiality of organizational communication.
• Mobile App Development: Designed, developed, and launched a polling and chat-based mobile app available on the
Play Store and App Store, receiving positive user feedback.
• Technologies Used: Node.js, Nest.js, JavaScript, TypeScript, Flask, Socket.io, RabbitMQ, MongoDB,
PostgreSQL, GraphQL, AWS (EKS, S3, CloudFront, Route53, CloudWatch), React.js, Dart, Flutter, Next.js,
Kubeshark, Redis, etc.
Senior Software Engineer (Contract) April 2022 – August 2022
Fairwords Inc. Longmont, CO, USA
• Maintained Legacy Systems: Assisted in maintaining and updating legacy systems written in .NET, ensuring their
stability and reliability.
• Development Contributions: Created change documentation, participated in code reviews, and conducted
integration and testing to enhance the development process.
• MVP Development: Collaborated with the Guide team to develop an MVP of their Guide Client product using
Electron.js.

• Guide App Enhancements: Improved user experience and functionality for the Guide App, a web application built
with AngularJS and Apollo Client, through regular updates and enhancements.
• Reusable Codebase: Developed reliable, reusable code suitable for deployment in distributed cloud environments,
improving scalability and adaptability.
• DevOps Innovations: Partnered with the DevOps team to implement solutions using the MongoDB Kafka
Connector and Kafka on Confluent Cloud, enhancing system performance and reliability.
• Technologies Used: Node.js, JavaScript, TypeScript, Lambda, Nest.js, GraphQL, AngularJS, Electron.js,
MongoDB, Kafka, RxJS, AWS, etc.
Tech Lead/Software Engineer June 2020 – September 2021
Xearth Pvt. Abidjan, Ivory Coast
• Developed Customer-Centric Applications: Contributed to developing applications with a strong focus on user
experience and satisfaction, leading to a 20% increase in customer retention.
• Infrastructure Management: Managed the infrastructure of all software products on AWS EKS, ensuring smooth
operations and high availability, resulting in a 40% reduction in infrastructure-related issues.
• Serverless Migration: Migrated services and applications to serverless technology, achieving significant cost savings
for customers and improving operational efficiency by 45%.
• System Re-Architecture: Proposed and led the re-architecture of existing systems to align with enterprise-grade
clean architecture and best practices, enhancing maintainability and scalability.
• Focus Pay Design: Designed and architected Focus Pay, a fintech application by Focus Group, which reached 100k
users and provided a seamless, secure user experience.
• Technologies Used: Node.js, JavaScript, TypeScript, Nest.js, Express.js, Flutter, Dart, Python, React Native,
GraphQL, etc.
Software Engineer September 2016 – July 2019
Asqii Llc Kumasi, Ghana
• School Management System: Developed and maintained a school management system used in three African
nations, improving education administration and operations. (https://schooldesk.cc).
• Notification System: Designed and built the notification system (Mimir) that powers the entire application,
enhancing communication and efficiency using Flask, Kafka, and third-party services.
• Customer Solutions: Collaborated with software development and testing teams to design and develop reliable
solutions that addressed customer needs for functionality, scalability, and performance.
• Desktop Application: Worked on an Electron React-based desktop application designed for schools using the
management system, providing an intuitive and seamless user experience.
• Technologies Used: Node.js, Express.js, Lambda, Python, Flask, Django, React, Electron.js, React Native,
PostgreSQL, GCP, JavaScript, TypeScript, Docker, Cloud Run, etc.
Projects
PP-Stream | Node js, nestjs, mongodb, webrtc, rabbitmq, flutter, dart, react js, socketio September 2022
• Built this live streaming application during a 24-hour hackathon, which supported many different formats mp4,
vimeo, etc. It also had a feature for replaying and pausing.
• It was built using an event-driven microservice architecture, with WebRTC and Socket. The technologies used
include node js, nestjs, mongodb, webrtc, rabbitmq, flutter, dart, react js, socketio, etc.
Fnf-express | Node.js, Express.js, NestJS March 2021
• Developed an NPM package to streamline the development of Node.js applications written in Express.js, inspired
by the NestJS CLI.
• Implemented MVC Architecture to enable easy server creation using a CLI-based approach.
Certifications
Certified AWS Cloud Practitioner
Certified AWS Solutions Architect Associate Level
Technical Skills
LanguagesC, C++, Python, Java, Dart, Javascript, and Typescript, C#, etc.
Frameworks: React, Nextjs, Angular, Flutter, Express Js, Django, Flask, Nest Js, Next Js, Angular Js, Electron Js,
Data: MySQL, Postgres, MongoDB, Sqlite, Cassandra, DynamoDB, ELK, Pinecone, etc.
Developer Tools: Git, Github, GCP, AWS, Trello, Slack, Discord, Jira, Confluence, Jenkins, Linux Server
Administration, Data-Dog, RabbitMQ, Kafka, Docker, Kubernetes, WebRTC, RMTP, XMPP, ELK, Grafanna,
Prometheus, Nginx, Apache, HAProxy, Cypress, Jest, Jmeter, etc
Libraries: pandas, numpy, scikitlearn, gensim, spacy, tensorflow, etc.
"""
predicted_skills = predict_skills(course_example)

print("Predicted skills:")
for skill, sim in predicted_skills:
    print(f"{skill}: {sim:.4f}")
