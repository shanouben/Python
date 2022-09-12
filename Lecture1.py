
from time import time
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


data = pd.read_excel('excel1.xlsx', sheet_name=1)
print(data)

dt =pd.to_datetime(data)
print(dt)