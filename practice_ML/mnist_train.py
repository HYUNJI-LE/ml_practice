from sklearn import model_selection, svm, metrics

def load_csv(fname):  #csv열고 가공
    labels = []
    images = []
    with open(fname, "r") as f:
        for line in f:
            cols = line.split(",")
            if len(cols) <2: continue
            labels.append(int(cols.pop(0)))
            vals = list(map(lambda n: int(n)/256,cols)) #255까지의 정수로 이뤄진 각 픽셀을 256으로 나눠 0이상이고 1미만인 실수 벡터가 됨
            images.append(vals)
    return {"labels":labels, "images":images}
data = load_csv("./mnist/train.csv")
test = load_csv("./mnist/t10k.csv")

#학습
clf= svm.SVC()
clf.fit(data["images"], data["labels"])

#예측
predict = clf.predict(test["images"])

#결과
ac_score = metrics.accuracy_score(test["labels"], predict)
cl_report = metrics.classification_report(test["labels"], predict)
print("정답률=", ac_score)
print("리포트=")
print(cl_report)




