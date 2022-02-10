from koeda import EDA
import pandas as pd
import sys
import time
from sklearn.model_selection import train_test_split
data = pd.read_csv("../../CloudData/math/data/train.csv")

eda = EDA(
    morpheme_analyzer="Okt", alpha_sr=0.0, alpha_ri=0.1, alpha_rs=0.0, prob_rd=0.0
)
print(data)
data_train = pd.DataFrame()
data_test = pd.DataFrame()
data_train["class"], data_test["class"], data_train["problem"], data_test["problem"], data_train["code"], data_test["code"], data_train["answer"], data_test["answer"] = \
    train_test_split(data["class"],data["problem"], data["code"], data["answer"], test_size=0.035, stratify=data['class'])
new1data = data_train
new2data = data_train
new3data = data_train

# print(data_train, len(data_train))
# print(data_test, len(data_test))

new1data['problem'] = new1data['problem'].apply(lambda x: eda(x))
eda = EDA(
    morpheme_analyzer="Okt", alpha_sr=0.0, alpha_ri=0.2, alpha_rs=0.0, prob_rd=0.0
)
new2data['problem'] = new1data['problem'].apply(lambda x: eda(x))
eda = EDA(
    morpheme_analyzer="Okt", alpha_sr=0.0, alpha_ri=0.3, alpha_rs=0.0, prob_rd=0.0
)
new3data['problem'] = new2data['problem'].apply(lambda x: eda(x))

data = pd.concat([data_train, new1data, new2data, new3data],axis=0)
# print(data)
# data.to_csv("../../CloudData/math/data/agutrain.csv")
# data_test.to_csv("../../CloudData/math/data/Valtrain.csv")

# data = pd.read_csv("../../CloudData/math/data/agutrain.csv")
# valdata = pd.read_csv("../../CloudData/math/data/Valtrain.csv")


data['problem'] = data['problem'].apply(lambda x: x + '<sys>') + data['class'].apply(lambda x: str(x) + '<sys>')
data_test['problem'] = data_test['problem'].apply(lambda x: x + '<sys>') + data_test['class'].apply(lambda x: str(x) + '<sys>')


# print(data)
data.to_csv("../../CloudData/math/data/Agutrain.csv")
data_test.to_csv("../../CloudData/math/data/Valtrain.csv")


