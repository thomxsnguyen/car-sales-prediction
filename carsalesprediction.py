import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns

#Import Dataset
car_df = pd.read_csv('Car_Purchasing_Data.csv', encoding = "ISO-8859-1")
print(car_df.tail(5))

