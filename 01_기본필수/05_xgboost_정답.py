"""실습 05: XGBoost도 같은 fit·predict 흐름으로 실행합니다."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# 제공: 이 파일 옆의 정상 CSV 20건을 읽습니다. 현재 터미널 위치와 무관합니다.
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "requests.csv")
develop, test = train_test_split(raw, test_size=0.2, stratify=raw["failed"], random_state=42)
train, valid = train_test_split(develop, test_size=0.25, stratify=develop["failed"], random_state=42)
# test 4건은 보관만 합니다. 입력은 실행 전에 아는 글자 수입니다.
X_train = train[["input_chars"]]
X_valid = valid[["input_chars"]]
y_train = train["failed"]
y_valid = valid["failed"]

from xgboost import XGBClassifier
from sklearn.metrics import f1_score

# 제공: 학습 회차 80·깊이 2·보정 크기 0.05를 고정합니다. CPU로 실행합니다.
model = XGBClassifier(
    n_estimators=80,             # 학습할 의사결정나무(Decision Tree)의 개수 (80개)
                                 # 이전 트리의 예측을 보완하는 트리를 순차적으로 추가
                                 # 값이 클수록 복잡한 패턴을 학습할 수 있지만 과적합 위험과 학습 시간 증가

    max_depth=2,                 # 각 의사결정나무의 최대 깊이 (2단계)
                                 # 값이 클수록 복잡한 데이터 패턴을 학습할 수 있음
                                 # 값이 너무 크면 과적합(Overfitting)이 발생할 수 있음

    learning_rate=0.05,          # 학습률: 새로운 트리의 보정값을 기존 예측에 반영하는 비율
                                 # 0.05는 각 트리의 보정값을 5% 크기로 반영한다는 의미
                                 # 값이 작을수록 점진적으로 학습하며, 일반적으로 더 많은 트리가 필요함

    objective="binary:logistic", # 학습 목적함수: 이진 분류(Binary Classification)
                                 # 정답이 0 또는 1인 문제에 사용
                                 # 로지스틱 손실을 최소화하며 클래스 1의 예측 확률을 학습

    eval_metric="logloss",       # 모델 평가 지표: 로그 손실(Log Loss)
                                 # 실제 정답과 예측 확률의 차이를 평가
                                 # 잘못된 예측에 높은 확신을 보일수록 큰 페널티 부여
                                 # 값이 0에 가까울수록 좋은 성능을 의미

    tree_method="hist",          # 트리 생성 방식: 히스토그램 기반 알고리즘
                                 # 연속형 특성값을 여러 구간(Bin)으로 나누어 분할 기준 탐색
                                 # 효율적인 계산을 통해 트리 학습 속도를 높이는 방식

    random_state=42,             # 난수 시드(Random Seed) 고정
                                 # 데이터 샘플링 등 무작위 동작의 재현성을 높이기 위한 설정
                                 # 42라는 숫자 자체에 특별한 의미는 없음

    n_jobs=1                     # 학습 과정에서 사용할 병렬 스레드 수
                                 # 1은 단일 스레드로 연산하도록 설정
                                 # -1은 사용 가능한 모든 CPU 논리 코어를 활용하도록 설정
)
# 해설 B05: train으로 학습하고 valid를 예측하세요.
# Random Forest와 학습 원리는 다르지만 사용 형태는 같습니다. eval_metric은 F1과 다른 진단 지표입니다.
model.fit(X_train, y_train)
prediction = model.predict(X_valid)
# 해설 끝
print("valid 예측:", prediction.tolist())
print("valid F1:", round(f1_score(y_valid, prediction, zero_division=0), 3))
print("학습 목적: 실패 분류 / 확인 지표: F1")
