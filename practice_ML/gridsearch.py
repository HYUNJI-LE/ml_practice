#GridSearch 사이킷런의 알고리즘들을 사용할 때 최적의 매개변수를 자동으로 찾아줌

import pandas as pd
from sklearn import model_selection, svm , metrics
from sklearn.model_selection import GridSearchCV

train_csv = pd.read_csv("./mnist_data/train.csv")
test_csv = pd.read_csv("./mnist_data/t10k.csv")

print(train_csv.info())
print(train_csv.head())
print(test_csv.info())
print(train_csv.head())

#필요한 열 추출하기
train_label = train_csv.iloc[:,0]
train_data = train_csv.iloc[:,1:577]
test_label = train_csv.iloc[:,0]
test_data = train_csv.iloc[:,1:577]
print("학습 데이터의 수  ", len(train_label))

#그리드서치 매개변수 설정
params = [{"C":[1,10,100,1000],"kernel":["linear"]},{"C":[1,10,100,1000],"kernel":["rbf"],"gamma":[0.001,0.0001]}]

#그리드서치 수행
clf = GridSearchCV(svm.SVC(), params, n_jobs=-1)#모델, 찾고자 하는 파라미터를 딕셔너리 형식으로 줌, 계산할 프로세스 수(-1은 전부)
clf.fit(train_data, train_label)
print("학습기", clf.best_estimator_)

#테스트데이터 확인하기
pre = clf.predict(test_data)
ac_score = metrics.accuracy_score(pre, test_label)
print("정답률", ac_score)