from sklearn.datasets import load_iris
import pandas as pd
import os

os.makedirs("data", exist_ok=True)

iris = load_iris(as_frame=True)
df = iris.frame
df.to_csv("data/iris.csv", index=False)

print("Saved dataset to data/iris.csv")