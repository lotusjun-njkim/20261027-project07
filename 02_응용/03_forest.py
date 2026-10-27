"""실습 03 응용: 같은 조건의 단일 트리와 숲을 비교합니다."""
from pathlib import Path
import pandas as pd

# 제공: 전날의 분할 소속뿐 아니라 position 순서까지 그대로 읽습니다.
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
manifest = pd.read_csv(HERE / "data/split_manifest.csv")
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"]
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"]
train = raw.loc[train_ids]
valid = raw.loc[valid_ids]
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
X_train, X_valid = train[FEATURES], valid[FEATURES]
y_train, y_valid = train["failed"], valid["failed"]
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
# test 20건의 입력·정답은 아래 모델에 전달하지 않습니다.
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# 제공: 전날 배운 결측 대체와 문자 변환입니다. 통계는 fit할 때만 배웁니다.
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), FEATURES[:2]),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), FEATURES[2:]),
])
# 트리는 이 실습에서 표준화하지 않습니다. 각 후보·fold마다 새 전처리기를 학습합니다.

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix

candidates = [("Tree", DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)),
              ("Forest", RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2,
                                    max_features="sqrt", random_state=42, n_jobs=1))]
rows = []
for name, estimator in candidates:
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    # TODO A03: model을 train으로 학습한 뒤 valid를 예측하세요.
    model.fit(X_train, y_train)
    prediction = model.predict(X_valid)
    # TODO 끝
#혼동행렬 
    tn, fp, fn, tp = confusion_matrix(y_valid, prediction, labels=[0, 1]).ravel()
    rows.append({"model": name, "train_F1": f1_score(y_train, model.predict(X_train), zero_division=0),
                 "valid_F1": f1_score(y_valid, prediction, zero_division=0), "FP": fp, "FN": fn})
result = pd.DataFrame(rows)
result.to_csv(OUT / "03_forest.csv", index=False, encoding="utf-8-sig")
print(result.round(3).to_string(index=False))



"""실습 03 응용: 같은 조건의 단일 트리와 숲을 비교합니다."""
from pathlib import Path
import pandas as pd

# 1. 고정된 데이터셋 및 실험 분할 명세(Manifest) 로드
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
manifest = pd.read_csv(HERE / "data/split_manifest.csv")

# 2. 사전에 약속된 train/valid 포지션 순서대로 ID를 정렬하여 인덱싱 추출
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"]
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"]

# 3. 데이터프레임에서 행을 추출하여 최종 학습(train) 및 검증(valid) 셋 생성
train = raw.loc[train_ids]
valid = raw.loc[valid_ids]

# 4. 사용할 입력 특성(Features)과 정답(failed) 분리
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
X_train, X_valid = train[FEATURES], valid[FEATURES]
y_train, y_valid = train["failed"], valid["failed"]

# 5. 결과 저장용 출력 폴더 생성
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

# 6. 사이킷런 전처리, 모델 뼈대 및 파이프라인 모듈 임포트
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# 7. 복합 데이터 전처리기(ColumnTransformer) 구성Blueprint 정의
pre = ColumnTransformer([
    # 숫자형 특성(인덱스 0, 1): 결측치를 중앙값(median)으로 대체
    ("num", SimpleImputer(strategy="median"), FEATURES[:2]),
    # 범주형 특성(인덱스 2, 3): 결측치를 최빈값으로 대체한 후 원-핫 인코딩 수행 파이프라인 연결
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), FEATURES[2:]),
])

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix

# 8. 실험 비교 대상 후보군 정의 (단일 트리 vs 랜덤 포레스트 숲)
candidates = [("Tree", DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)),
              ("Forest", RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2,
                                    max_features="sqrt", random_state=42, n_jobs=1))]
rows = []

# 9. 반복문을 통한 모델 순차 학습 및 교차 평가
for name, estimator in candidates:
    # 매 루프마다 전처리기 구조를 깨끗하게 복사(clone)하여 모델 예측기와 결합한 파이프라인 생성
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    
    # [TODO A03]: model을 train으로 학습한 뒤 valid를 예측하세요.
    model.fit(X_train, y_train)       # 훈련 데이터로 전처리 통계 학습 및 분류기 동시 훈련
    prediction = model.predict(X_valid) # 검증 데이터 전처리 변환 및 정답 라벨 예측
    # TODO 끝
    
    # 10. 오차행렬(Confusion Matrix) 계산 및 평탄화(.ravel()) 작업 진행
    tn, fp, fn, tp = confusion_matrix(y_valid, prediction, labels=[0, 1]).ravel()
    
    # 11. 평가 결과 지표(F1-score, FP, FN)를 딕셔너리 형태로 리스트에 누적
    rows.append({"model": name, "train_F1": f1_score(y_train, model.predict(X_train), zero_division=0),
                 "valid_F1": f1_score(y_valid, prediction, zero_division=0), "FP": fp, "FN": fn})

# 12. 결과를 데이터프레임으로 변환 후 CSV 저장 및 콘솔창 출력
result = pd.DataFrame(rows)
result.to_csv(OUT / "03_forest.csv", index=False, encoding="utf-8-sig")
print(result.round(3).to_string(index=False))
