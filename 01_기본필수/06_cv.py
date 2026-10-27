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
folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
model = DecisionTreeClassifier(max_depth=3, random_state=42)
# TODO B06: model을 train에서 3-fold F1으로 평가하세요.
scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="f1")
# TODO 끝
print("fold별 F1:", scores.round(3).tolist())
print("평균 F1:", round(scores.mean(), 3))
print("표준편차:", round(scores.std(), 3))






#"""실습 06: train 12건 안에서만 3-fold 교차검증합니다."""
#from pathlib import Path
#import pandas as pd
#from sklearn.model_selection import train_test_split
#
## [경로 설정] 스크립트 파일이 위치한 현재 디렉터리의 절대 경로를 자동으로 찾습니다.
#HERE = Path(__file__).resolve().parent
#
## [데이터 로드] 현재 파일과 같은 위치에 있는 requests.csv 파일을 읽어 데이터프레임으로 변환합니다.
#raw = pd.read_csv(HERE / "requests.csv")
#
## [데이터 분할 1단계] 전체 데이터 중 80%는 모델 개발용(develop), 20%는 최종 평가 보관용(test, 4건)으로 먼저 분할합니다.
#develop, test = train_test_split(raw, test_size=0.2, stratify=raw["failed"], random_state=42)
#
## [데이터 분할 2단계] 개발용 데이터를 다시 실제 학습용(train, 12건)과 내부 검증용(valid, 4건)으로 나눕니다.
#train, valid = train_test_split(develop, test_size=0.25, stratify=develop["failed"], random_state=42)
#
## test 4건은 보관만 합니다. 입력은 실행 전에 아는 글자 수입니다.
## [X, y 분리] 12건의 학습 데이터셋에서 입력 피처(input_chars)와 정답 타겟(failed)을 각각 추출합니다.
#X_train = train[["input_chars"]]
#X_valid = valid[["input_chars"]]
#y_train = train["failed"]
#y_valid = valid["failed"]
#
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.model_selection import StratifiedKFold, cross_val_score
#
## [교차검증 분할기 생성] 정답(failed) 클래스 비율을 균등하게 유지하면서 데이터를 3개의 덩어리(Fold)로 나눌 분할기 객체를 만듭니다.
#folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
#
## [모델 생성] 최대 깊이를 3으로 제한한 의사결정나무(Decision Tree) 모델을 인스턴스화합니다.
#model = DecisionTreeClassifier(max_depth=3, random_state=42)
#
## [TODO B06] 12건의 train 데이터 내부에서 3-fold 세트를 돌려가며 F1-score 점수를 계산합니다.
#scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="f1")
## TODO 끝
#
## [결과 출력] 각 Fold별로 나온 3개의 F1 점수를 리스트로 출력하고, 이들의 평균값과 표준편차(점수 변동성)를 계산해 보여줍니다.
#print("fold별 F1:", scores.round(3).tolist())
#print("평균 F1:", round(scores.mean(), 3))
#print("표준편차:", round(scores.std(), 3))
#