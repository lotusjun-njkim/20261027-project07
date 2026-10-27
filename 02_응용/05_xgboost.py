"""실습 05 응용: PPT와 같은 XGBoost 80회차 분류 모델입니다."""
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

from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix, log_loss

estimator = XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist", device="cpu", random_state=42, n_jobs=1)
model = Pipeline([("pre", clone(pre)), ("model", estimator)])
# TODO A05: Pipeline을 학습하고 valid를 예측하세요.
pass
prediction = None
# TODO 끝
probability = model.predict_proba(X_valid)[:, 1]  # classes_ = [0, 1] 확인 후 실패1의 열 사용
print("클래스 순서:", model.classes_.tolist())
tn, fp, fn, tp = confusion_matrix(y_valid, prediction, labels=[0, 1]).ravel()
result = pd.DataFrame([{"train_F1": f1_score(y_train, model.predict(X_train), zero_division=0),
                       "valid_F1": f1_score(y_valid, prediction, zero_division=0), "FP": fp, "FN": fn,
                       "valid_logloss": log_loss(y_valid, probability, labels=[0, 1])}])
result.to_csv(OUT / "05_xgboost.csv", index=False, encoding="utf-8-sig")
print(result.round(3).to_string(index=False))
print("objective는 학습 목적, logloss는 진단용, 후보 비교의 주 지표는 F1입니다.")



"""실습 05 응용: PPT와 같은 XGBoost 80회차 분류 모델입니다."""
from pathlib import Path
import pandas as pd

# [경로 설정] 스크립트 파일이 위치한 현재 디렉터리의 절대 경로를 자동으로 찾습니다.
HERE = Path(__file__).resolve().parent

# [데이터 로드] 원본 요청 데이터를 읽어오면서 'request_id' 컬럼을 행의 고유 이름(인덱스)으로 지정합니다.
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")

# [명단 로드] 미리 분할해 둔 훈련용/검증용 데이터의 명단(인덱스 및 순서 정보) 파일을 읽어옵니다.
manifest = pd.read_csv(HERE / "data/split_manifest.csv")

# [ID 추출] 명단에서 'train'과 'valid'에 해당하는 행만 골라내고, 원래 저장된 순서(position)대로 정렬하여 ID를 추출합니다.
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"]
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"]

# [데이터 분할] 추출한 ID를 기반으로 원본 데이터셋에서 실제 학습용 및 검증용 데이터를 행 단위로 쏙 골라냅니다.
train = raw.loc[train_ids]
valid = raw.loc[valid_ids]

# [피처 지정] 모델 학습에 사용할 4개의 입력 특성(피처) 이름을 리스트로 정의합니다.
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]

# [X, y 분리] 정의한 피처들을 기준으로 입력 데이터(X)와 맞추어야 하는 정답 타겟(y, 실패 여부)을 각각 분리합니다.
X_train, X_valid = train[FEATURES], valid[FEATURES]
y_train, y_valid = train["failed"], valid["failed"]

# [폴더 생성] 실험 결과를 저장할 'outputs' 폴더를 생성합니다. (이미 존재하면 에러 없이 넘어감)
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

# test 20건의 입력·정답은 아래 모델에 전달하지 않습니다.
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# [전처리 설계] 수치형 데이터와 범주형(문자형) 데이터에 각각 서로 다른 전처리를 병렬로 적용하는 변환기를 만듭니다.
pre = ColumnTransformer([
    # 수치형 피처(글자 수 등 앞의 2개): 빈칸을 데이터의 '중간값(median)'으로 채웁니다.
    ("num", SimpleImputer(strategy="median"), FEATURES[:2]),
    # 범주형 피처(작업 유형 등 뒤의 2개): 빈칸을 '최빈값'으로 채운 뒤, 문자열을 0과 1의 디지털 배열로 원-핫 인코딩합니다.
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), FEATURES[2:]),
])
# 트리는 이 실습에서 표준화하지 않습니다. 각 후보·fold마다 새 전처리기를 학습합니다.

from xgboost import XGBClassifier
from sklearn.metrics import f1_score, confusion_matrix, log_loss

# [모델 생성] 제공된 하이퍼파라미터 조건으로 XGBoost 이진 분류기 모델을 인스턴스화합니다.
estimator = XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist", device="cpu", random_state=42, n_jobs=1)

# [파이프라인 구축] 전처리 전파 차단을 위해 pre 구조를 복사(clone)한 후, XGBoost 모델과 파이프라인으로 연결합니다.
model = Pipeline([("pre", clone(pre)), ("model", estimator)])

# [TODO A05] Pipeline을 학습하고 valid를 예측하세요.
model.fit(X_train, y_train)          # 파이프라인 전체를 구동하여 전처리 계산 후 XGBoost를 학습시킵니다.
prediction = model.predict(X_valid)  # 학습 완료된 모델을 통해 검증용 데이터의 클래스(0 또는 1)를 예측합니다.
# TODO 끝

# [확률 예측] 검증 데이터가 '실패(1)'일 확률(0~1 사이값)을 계산합니다. classes_=[0, 1] 순서이므로 인덱스 1번 열을 가져옵니다.
probability = model.predict_proba(X_valid)[:, 1]  

# [클래스 순서 확인] 모델 내부적으로 0과 1이 어떤 순서로 정렬되어 인덱싱되었는지 리스트로 출력합니다.
print("클래스 순서:", model.classes_.tolist())

# [혼동 행렬 계산] 예측 성능 진단을 위해 오차 행렬을 구하고, 이를 1차원 배열로 평탄화하여 4개의 평가지표로 분할합니다.
# tn: 정상 예측 성공, fp: 정상을 실패로 오판, fn: 실패를 정상으로 오판, tp: 실패 예측 성공
tn, fp, fn, tp = confusion_matrix(y_valid, prediction, labels=[0, 1]).ravel()

# [결과 기록] 훈련/검증 F1-score, 오판 수(FP, FN), 그리고 예측 확률의 정밀도를 기록할 로그 손실(logloss)을 데이터프레임에 담습니다.
result = pd.DataFrame([{"train_F1": f1_score(y_train, model.predict(X_train), zero_division=0),
                       "valid_F1": f1_score(y_valid, prediction, zero_division=0), "FP": fp, "FN": fn,
                       "valid_logloss": log_loss(y_valid, probability, labels=[0, 1])}])

# [결과 저장] 수집된 통합 실험 결과 표를 '05_xgboost.csv' 파일로 안전하게 저장합니다. (한글 깨짐 방지 적용)
result.to_csv(OUT / "05_xgboost.csv", index=False, encoding="utf-8-sig")

# [출력] 화면에 소수점 3자리까지 보기 좋게 정리된 텍스트 표로 실험 결과를 출력합니다.
print(result.round(3).to_string(index=False))
print("objective는 학습 목적, logloss는 진단용, 후보 비교의 주 지표는 F1입니다.")
