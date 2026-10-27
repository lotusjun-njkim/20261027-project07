"""실습 01: 입력을 질문으로 나누는 트리를 만듭니다. 성능 평가가 아닌 원리 예제입니다."""
from sklearn.tree import DecisionTreeClassifier

X = [[100], [200], [300], [700], [800], [900]]
y = [0, 0, 0, 1, 1, 1]  # 같은 순서의 정답: 성공 0, 실패 1
model = DecisionTreeClassifier(max_depth=1, random_state=42)
# 해설 B01a: X와 y로 model을 학습하세요.
# fit은 입력과 정답을 함께 사용해 분기 기준을 찾습니다. 아직 예측을 하는 문장이 아닙니다.
model.fit(X, y)
# 해설 끝
# 해설 B01b: 250과 850을 예측하여 prediction에 넣으세요.
# predict에는 정답을 주지 않습니다. [[250], [850]]은 2행 1열 입력입니다.
prediction = model.predict([[250], [850]])
# 해설 끝
print("분기 기준:", model.tree_.threshold[0])
print("250·850 예측:", prediction.tolist())
