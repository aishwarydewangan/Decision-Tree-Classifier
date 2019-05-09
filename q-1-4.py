import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

tips = pd.read_csv("data.csv")
ax = sns.scatterplot(x="satisfaction_level", y="number_project",hue="left", style="left", data=tips)
plt.show()