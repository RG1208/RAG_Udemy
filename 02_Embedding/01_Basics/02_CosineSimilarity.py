import numpy

#Measuring Cosine similarity 

def cosine_similarity(vec1,vec2):
    dot_product=numpy.dot(vec1,vec2)
    norm_vec1=numpy.linalg.norm(vec1)
    norm_vec2=numpy.linalg.norm(vec2)
    return dot_product/(norm_vec1*norm_vec2)

cat=[0.8,0.9]
dog=[0.1,0.5]
kitten=[0.7,0.9]
bitch=[0.05,0.5]
car=[0.5,0.1]
truck=[0.4,0.2]

cat_Kitten=cosine_similarity(cat,kitten) 
cat_Dog=cosine_similarity(cat,dog)
cat_Car=cosine_similarity(cat,car)

print(f"Cosine Similarity between 'cat' and 'kitten': {cat_Kitten:.4f}")
print(f"Cosine Similarity between 'cat' and 'Dog': {cat_Dog:.4f}")
print(f"Cosine Similarity between 'cat' and 'Car': {cat_Car:.4f}")

