import sys
import numpy
import sklearn_extra

print("--- DIAGNOSTICS ---")
print(f"Python Executable: {sys.executable}")
print(f"NumPy Version: {numpy.__version__}")
print(f"NumPy Location: {numpy.__file__}")
print(f"sklearn_extra Location: {sklearn_extra.__file__}")
print("---------------------\n")

# Your original code starts here...
from sklearn_extra.cluster import KMedoids
# ... rest of your script