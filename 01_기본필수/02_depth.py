"""실습 02: 깊이만 바꾸고 같은 train·valid에서 비교합니다."""
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
from sklearn.metrics import f1_score

for depth in [1, 3]:
    # TODO B02: 반복 중인 depth를 max_depth에 연결하세요.
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    # TODO 끝
    model.fit(X_train, y_train)
    train_f1 = f1_score(y_train, model.predict(X_train), zero_division=0)
    valid_f1 = f1_score(y_valid, model.predict(X_valid), zero_division=0)
    print(f"깊이 {depth}: train F1={train_f1:.3f}, valid F1={valid_f1:.3f}")

"""실습 02: 깊이만 바꾸고 같은 train·valid에서 비교합니다."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 파일 경로 설정 및 데이터 로드 (현재 스크립트 파일이 위치한 폴더의 requests.csv 읽기)
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "requests.csv")

# 2. 데이터를 Train(60%), Valid(20%), Test(20%) 비율로 분할 (계층적 분할 적용)
develop, test = train_test_split(raw, test_size=0.2, stratify=raw["failed"], random_state=42)
train, valid = train_test_split(develop, test_size=0.25, stratify=develop["failed"], random_state=42)

# 3. 학습 및 검증에 사용할 독립변수(X: 글자 수)와 종속변수(y: 실패 여부) 추출
X_train = train[["input_chars"]]
X_valid = valid[["input_chars"]]
y_train = train["failed"]
y_valid = valid["failed"]

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

# 4. 트리의 최대 깊이(depth)를 1과 3으로 바꾸어가며 반복 학습 및 평가
for depth in:
    # TODO B02: 반복 중인 depth를 max_depth에 연결하세요.
    # 반복문에서 제공하는 depth 변수(1 또는 3)를 모델의 max_depth 파라미터에 대입합니다.
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    # TODO 끝
    
    # 모델 학습 (훈련 데이터 활용)
    model.fit(X_train, y_train)
    
    # 훈련 데이터와 검증 데이터 각각에 대한 예측값 생성 및 F1-score 측정
    train_f1 = f1_score(y_train, model.predict(X_train), zero_division=0)
    valid_f1 = f1_score(y_valid, model.predict(X_valid), zero_division=0)
    
    # 각 깊이별 최종 성능 출력
    print(f"깊이 {depth}: train F1={train_f1:.3f}, valid F1={valid_f1:.3f}")





"""실습 02: 깊이만 바꾸고 같은 train·valid에서 비교합니다."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 파일 경로 설정 및 데이터 로드 (현재 스크립트 파일이 위치한 폴더의 requests.csv 읽기)
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "requests.csv")

# 2. 데이터를 Train(60%), Valid(20%), Test(20%) 비율로 분할 (계층적 분할 적용)
develop, test = train_test_split(raw, test_size=0.2, stratify=raw["failed"], random_state=42)
train, valid = train_test_split(develop, test_size=0.25, stratify=develop["failed"], random_state=42)

# 3. 학습 및 검증에 사용할 독립변수(X: 글자 수)와 종속변수(y: 실패 여부) 추출
X_train = train[["input_chars"]]
X_valid = valid[["input_chars"]]
y_train = train["failed"]
y_valid = valid["failed"]

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

# 4. 트리의 최대 깊이(depth)를 1과 3으로 바꾸어가며 반복 학습 및 평가
for depth in:
    # TODO B02: 반복 중인 depth를 max_depth에 연결하세요.
    # 반복문에서 제공하는 depth 변수(1 또는 3)를 모델의 max_depth 파라미터에 대입합니다.
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    # TODO 끝
    
    # 모델 학습 (훈련 데이터 활용)
    model.fit(X_train, y_train)
    
    # 훈련 데이터와 검증 데이터 각각에 대한 예측값 생성 및 F1-score 측정
    train_f1 = f1_score(y_train, model.predict(X_train), zero_division=0)
    valid_f1 = f1_score(y_valid, model.predict(X_valid), zero_division=0)
    
    # 각 깊이별 최종 성능 출력
    print(f"깊이 {depth}: train F1={train_f1:.3f}, valid F1={valid_f1:.3f}")
