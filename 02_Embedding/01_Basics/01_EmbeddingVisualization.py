import numpy
import matplotlib.pyplot as plt

word_embeddings={
    "cat":[0.8,0.9],
    "dog":[0.1,0.5],
    "kitten":[0.7,0.9],
    "bitch":[0.05,0.5],
    "car":[0.5,0.1],
    "truck":[0.4,0.2],
}

fig, ax =plt.subplots(figsize=(8,6))

for word, coords in word_embeddings.items():
    ax.scatter(coords[0],coords[1],s=100)
    ax.annotate(word,
                (coords[0], coords[1]),
                xytext=(5,5),
                textcoords="offset points")
    
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title("simplified word embeddings in 2D space")
    ax.grid(True,alpha=0.3)

plt.tight_layout()
plt.show()
    
