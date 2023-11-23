import torch

x = torch.arange(0, 20).reshape(5, 4).float()

print(x)
def dist(x):
    # x is a vector that contains pairs of vectors concatenated
    # We want to split it into pairs of vectors and compute the distance between them
    d = 1 - torch.nn.functional.cosine_similarity(x[:, :x.shape[1]//2], x[:, x.shape[1]//2:]).reshape(-1, 1)
    
    return d

d = dist(x)
print(f"sqrt2: {8**0.5}")
print(d.shape, d)