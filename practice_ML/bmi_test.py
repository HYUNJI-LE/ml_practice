from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

tbl = pd.read_csv("bmi.csv")

label = tbl["label"]
w=tbl["weight"]/100 #최대 100kg라고 가정
h=tbl["height"]/200
wh = pd.concat([w,h],axis=1)

data_train, data_test, label_train, label_test = train_test_split(wh,label)

clf = svm.SVC()
clf.fit(data_train, label_train)

predict = clf.predict(data_test)

ac_score = metrics.accuracy_score(label_test, predict)
cl_report = metrics.classification_report(label_test, predict)

print("정답률 = ", ac_score)
print("리포트 =\n", cl_report)

tbl = pd.read_csv("bmi.csv", index_col=2)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

def scatter(lbl, color): #서브 플롯 전용 - 지정한 레이블의 임의의 색으로 칠하기 (tjqmvmffhtdmf tkdydgo tordmf rnqns)
    b = tbl.loc[lbl]
    ax.scatter(b["weight"],b["height"], c=color, label=lbl)

scatter("fat", "red")
scatter("normal", "yellow")
scatter("thin", "purple")

ax.legend()
plt.savefig("bmi-test.png")
plt.show()