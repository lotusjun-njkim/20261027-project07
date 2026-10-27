# """실습 06 응용: Pipeline 전체를 같은 train 5-fold로 평가합니다."""
# from pathlib import Path
# import pandas as pd

# # 제공: 전날의 분할 소속뿐 아니라 position 순서까지 그대로 읽습니다.
# HERE = Path(__file__).resolve().parent
# raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
# manifest = pd.read_csv(HERE / "data/split_manifest.csv")
# train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"]
# valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"]
# train = raw.loc[train_ids]
# valid = raw.loc[valid_ids]
# FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
# X_train, X_valid = train[FEATURES], valid[FEATURES]
# y_train, y_valid = train["failed"], valid["failed"]
# OUT = HERE / "outputs"
# OUT.mkdir(exist_ok=True)
# # test 20건의 입력·정답은 아래 모델에 전달하지 않습니다.
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.base import clone

# # 제공: 전날 배운 결측 대체와 문자 변환입니다. 통계는 fit할 때만 배웁니다.
# pre = ColumnTransformer([
#     ("num", SimpleImputer(strategy="median"), FEATURES[:2]),
#     ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
#                       ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), FEATURES[2:]),
# ])
# # 트리는 이 실습에서 표준화하지 않습니다. 각 후보·fold마다 새 전처리기를 학습합니다.

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
# from sklearn.model_selection import StratifiedKFold, cross_val_score

# folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train))
# candidates = [("Tree", DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)), ("Forest", RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2,
#                                     max_features="sqrt", random_state=42, n_jobs=1)), ("XGB", XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.05,
#                   subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
#                   objective="binary:logistic", eval_metric="logloss",
#                   tree_method="hist", device="cpu", random_state=42, n_jobs=1))]
# rows = []
# for name, estimator in candidates:
#     model = Pipeline([("pre", clone(pre)), ("model", estimator)])
#     # TODO A06: 전처리까지 포함한 model의 fold별 F1을 계산하세요.
#     scores = None
#     # TODO 끝
#     print(name, "fold F1:", scores.round(4).tolist(), "평균:", round(scores.mean(), 4), "표준편차:", round(scores.std(), 4))
#     for number, value in enumerate(scores, start=1):
#         rows.append({"model": name, "fold": number, "fit_n": 48, "evaluate_n": 12, "F1": value})
# pd.DataFrame(rows).to_csv(OUT / "06_cv_scores.csv", index=False, encoding="utf-8-sig")
# print("사용 자료: train60 내부만 / valid20·test20 미사용")





"""실습 06 응용: Pipeline 전체를 같은 train 5-fold로 평가합니다."""
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

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# [교차검증 인덱스 생성] 동일한 조건에서 비교할 수 있도록 고정된 5-Fold 훈련/검증용 행 인덱스 세트 리스트를 생성합니다.
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train))

# [비교 모델 리스트 정의] 성능을 비교 실험할 3가지 머신러닝 알고리즘 후보를 이름과 설정값 쌍으로 구성합니다.
candidates = [
    # 1. 단일 의사결정나무 모델
    ("Tree", DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)), 
    # 2. 배깅 기반 앙상블 모델인 랜덤 포레스트
    ("Forest", RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=1)), 
    # 3. 부스팅 기반 강력한 앙상블 모델인 XGBoost
    ("XGB", XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cpu", random_state=42, n_jobs=1))
]

rows = []
# [실험 루프] 세 가지 모델 후보를 순서대로 하나씩 꺼내어 교차 검증을 시작합니다.
for name, estimator in candidates:
    # [파이프라인 구축] 매번 깨끗하게 복사한 전처리기 구조와 머신러닝 모델을 파이프라인 하나로 결합합니다.
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    
    # [TODO A06] 전처리 단계까지 완전히 격리 및 포함한 전체 파이프라인의 5-Fold F1 점수 배열을 자동 계산합니다.
    scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="f1")
    # TODO 끝
    
    # [결과 출력] 터미널 화면에 현재 모델의 각 Fold별 점수 리스트와 평균값, 그리고 표준편차를 소수점 4자리까지 출력합니다.
    print(name, "fold F1:", scores.round(4).tolist(), "평균:", round(scores.mean(), 4), "표준편차:", round(scores.std(), 4))
    
    # [데이터 수집] 수집된 5번의 점수를 반복문을 돌며 상세 정보(모델명, 폴드 번호, 학습 데이터 수 등)와 함께 기록 리스트에 추가합니다.
    for number, value in enumerate(scores, start=1):
        rows.append({"model": name, "fold": number, "fit_n": 48, "evaluate_n": 12, "F1": value})

# [결과 내보내기] 저장된 결과를 데이터프레임 구조로 변환한 후 '06_cv_scores.csv' 파일로 저장합니다.
pd.DataFrame(rows).to_csv(OUT / "06_cv_scores.csv", index=False, encoding="utf-8-sig")
print("사용 자료: train60 내부만 / valid20·test20 미사용")
