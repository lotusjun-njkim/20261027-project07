"""실습 01: 입력을 질문으로 나누는 트리를 만듭니다. 성능 평가가 아닌 원리 예제입니다."""
from sklearn.tree import DecisionTreeClassifier

X = [[100], [200], [300], [700], [800], [900]]
y = [0, 0, 0, 1, 1, 1]  # 같은 순서의 정답: 성공 0, 실패 1
model = DecisionTreeClassifier(max_depth=1, random_state=42)
# TODO B01a: X와 y로 model을 학습하세요.
model.fit(X,y)
# TODO 끝
# TODO B01b: 250과 850을 예측하여 prediction에 넣으세요.
prediction = model.predict([[250], [850], [1200]])
# TODO 끝
print("분기 기준:", model.tree_.threshold[0])
print("250·850·1200 예측:", prediction.tolist())





"""실습 01: 입력을 질문으로 나누는 트리를 만듭니다. 성능 평가가 아닌 원리 예제입니다."""
from sklearn.tree import DecisionTreeClassifier

# 1. 학습에 사용할 입력 데이터(특성)와 정답(레이블) 정의
X = [[100], [200], [300], [700], [800], [900]]  # 입력 데이터 (2차원 리스트 형태)
y = [0, 0, 0, 1, 1, 1]  # 같은 순서의 정답: 성공 0, 실패 1

# 2. 의사결정나무 모델 객체 생성 (최대 깊이를 1로 설정하여 질문을 딱 한 번만 던지도록 제한)
model = DecisionTreeClassifier(max_depth=1, random_state=42)

# TODO B01a: X와 y로 model을 학습하세요.
# fit() 함수를 사용하여 모델에게 준비된 데이터와 정답을 가르칩니다.
model.fit(X, y)
# TODO 끝

# TODO B01b: 250과 850을 예측하여 prediction에 넣으세요.
# predict() 함수에 예측하고 싶은 데이터를 2차원 형태로 전달합니다.
prediction = model.predict([[250], [850]])
# TODO 끝

# 3. 모델이 찾아낸 분기 기준점과 새로운 데이터에 대한 예측 결과 출력
print("분기 기준:", model.tree_.threshold[0])
print("250·850 예측:", prediction.tolist())
