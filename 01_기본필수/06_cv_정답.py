"""실습 06: train 12건 안에서만 3-fold 교차검증합니다."""
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

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 제공: 각 클래스 수를 고려해 기본필수는 3-fold입니다. valid4·test4는 넣지 않습니다.
folds = StratifiedKFold(
    n_splits=3,         # 전체 데이터를 3개의 Fold(부분 집합)로 분할
                        # 3번의 교차검증을 수행하며, 매번 2개 Fold는 학습용,
                        # 나머지 1개 Fold는 검증용으로 사용
                        # 각 Fold가 한 번씩 검증 데이터로 사용됨

    shuffle=True,       # 데이터를 Fold로 나누기 전에 무작위로 섞음
                        # 원본 데이터의 정렬 순서가 분할 결과에 미치는 영향을 줄임
                        # 단, 각 클래스의 비율은 최대한 유지하도록 분할

    random_state=42     # 데이터 셔플에 사용할 난수 시드 고정
                        # 코드를 반복 실행해도 동일한 Fold 구성이 생성되도록 설정
                        # shuffle=True일 때 적용됨
)
model = DecisionTreeClassifier(max_depth=3, random_state=42)
# 해설 B06: model을 train에서 3-fold F1으로 평가하세요.
# cross_val_score는 매 fold마다 새 모델을 학습합니다. scores는 3개의 평가 점수를 가진 배열입니다.
scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="f1")
# 해설 끝
print("fold별 F1:", scores.round(3).tolist())
print("평균 F1:", round(scores.mean(), 3))
print("표준편차:", round(scores.std(), 3))
