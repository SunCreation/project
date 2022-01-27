import pandas as pd
import re
data = pd.read_csv("/workspace/CloudData/math/data/train.csv")

for i in data["problem"]:
    re.findall("", i)