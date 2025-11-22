import os
import pandas
from typing import List,Dict,Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
print("setup complete")

doc=Document(
    page_content="This is a sample document.",
    metadata= {
        "source": "sample_source",
        "page":1,
        "author":"Rachit Garg",
        "date_created":"2025-22-11"
        }
)
print("Document created:")

import os
import pandas
from typing import List,Dict,Any

os.makedirs("data/text_files/", exist_ok=True)

sample_texts={
    "data/text_files/doc1.txt":"""
Introduction to Python

Python is a high-level, general-purpose programming language known for its simplicity, readability, and powerful capabilities. Created by Guido van Rossum and first released in 1991, Python has become one of the most popular programming languages in the world, used by beginners and professionals alike.

Python’s clean and easy-to-understand syntax makes it ideal for learning programming concepts. At the same time, its rich ecosystem of libraries and frameworks allows developers to build a wide range of applications — from small scripts to large-scale systems.

Python is widely used in:
    1. Web Development (Django, Flask)
    2. Data Science & Machine Learning (NumPy, Pandas, Scikit-learn, TensorFlow)
    3. Automation & Scripting
    4. Artificial Intelligence
    5. Scientific Computing
    6. Cybersecurity
    7. Game Development
    8. IoT & Embedded Systems

Python follows a philosophy known as “The Zen of Python,” which emphasizes writing code that is simple, readable, and maintainable. Because of its versatility and beginner-friendly nature, Python has become a top choice for students, developers, researchers, and organizations across industries.

In short, Python is a language that helps you think clearly, work efficiently, and build powerful solutions with ease.
""",

"data/text_files/doc2.txt":"""
Introduction to Machine Learning
Machine Learning (ML) is a branch of Artificial Intelligence that enables computers to learn patterns from data and make predictions or decisions without being explicitly programmed. Instead of following fixed rules, ML models improve their performance by analyzing examples, making it useful for tasks like classification, prediction, recommendation, and pattern detection. ML generally includes supervised learning (learning from labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through rewards). It follows a workflow of collecting data, training models, evaluating performance, and improving accuracy, and is widely used in fields such as healthcare, finance, automation, and technology.
"""
}

for file_path, content in sample_texts.items():
    with open(file_path, "w",encoding="utf-8") as f:
        f.write(content.strip())
print("Sample text files created.")
