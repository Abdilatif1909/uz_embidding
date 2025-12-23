import numpy as np

def cosine_sim(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return np.dot(v1, v2) / denom
