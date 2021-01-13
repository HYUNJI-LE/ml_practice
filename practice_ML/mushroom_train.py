import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

mr = pd.read_csv("./mushroom.csv", header=None)

print(mr.head())

label = []
data = []
attr_list = []

for row_index, row in mr.iterrows():
    label.append(row.loc[0])
    row_data = []
    for v in row.loc[1:]:
        row_data.append(ord(v)) #ord 문자의 아스키값을 돌려주는 함수
    data.append(row_data)

#테스트와 학습용 데이터 분리
data_train, data_test, label_train,label_test = train_test_split(data, label)

clf = RandomForestClassifier()
clf.fit(data_train, label_train)

predict = clf.predict(data_test)

ac_score = metrics.accuracy_score(label_test, predict)
cl_report = metrics.classification_report(label_test, predict)
print("정답률 =", ac_score)
print("리포트 =\n", cl_report)