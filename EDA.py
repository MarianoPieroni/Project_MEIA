import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df =pd.read_csv("imdb-videogames.csv")

print("DataSet")
print(f"\ndimensoes{df.shape}")
print(f"\ncolunas\n{df.columns}")
print(f"\nvalores nulos\n{df.isnull().sum()}")
print(f"total de nulos\n{df.isnull().sum().sum()}")
print(f"\nvalores duplicados\n{df.duplicated().sum()}")
print(f"\ntipos de dados\n{df.dtypes}")