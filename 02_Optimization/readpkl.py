import pickle
import os

with open('antenna_cache.pkl', 'rb') as f:
    data = pickle.load(f)
    
    
print("Cache loaded successfully.")
print(f"Cache contains {len(data)} entries.")
