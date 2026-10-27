# """실습 07~08 응용: 얕은 폴더 구조에서 분류 실험표를 완성합니다."""
# from pathlib import Path
# import pandas as pd
# 
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
# 
# # 제공: 전날 배운 결측 대체와 문자 변환입니다. 통계는 fit할 때만 배웁니다.
# pre = ColumnTransformer([
#     ("num", SimpleImputer(strategy="median"), FEATURES[:2]),
#     ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
#                       ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), FEATURES[2:]),
# ])
# # 트리는 이 실습에서 표준화하지 않습니다. 각 후보·fold마다 새 전처리기를 학습합니다.
# 
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.dummy import DummyClassifier
# from xgboost import XGBClassifier
# from sklearn.model_selection import StratifiedKFold, cross_val_score
# from sklearn.metrics import f1_score, confusion_matrix
# 
# folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train))
# candidates = [("Dummy", DummyClassifier(strategy="most_frequent")),
#               ("Tree", DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)), ("Forest", RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2,
#                                     max_features="sqrt", random_state=42, n_jobs=1)), ("XGB", XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.05,
#                   subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
#                   objective="binary:logistic", eval_metric="logloss",
#                   tree_method="hist", device="cpu", random_state=42, n_jobs=1))]
# rows, cv_rows = [], []
# predictions = pd.DataFrame({"request_id": X_valid.index, "actual": y_valid.to_numpy()})
# for name, estimator in candidates:
#     model = Pipeline([("pre", clone(pre)), ("model", estimator)])
#     # TODO A07a: train 내부 교차검증 뒤 전체 train으로 학습하세요.
#     scores = None
#     pass
#     # TODO 끝
#     # TODO A07b: valid의 예측을 prediction에 넣으세요.
#     prediction = None
#     # TODO 끝
#     tn, fp, fn, tp = confusion_matrix(y_valid, prediction, labels=[0, 1]).ravel()
#     rows.append({"model": name, "CV_F1": scores.mean(), "CV_std": scores.std(),
#                  "train_F1": f1_score(y_train, model.predict(X_train), zero_division=0),
#                  "valid_F1": f1_score(y_valid, prediction, zero_division=0), "FP": fp, "FN": fn})
#     for number, value in enumerate(scores, start=1):
#         cv_rows.append({"model": name, "fold": number, "fit_n": 48, "evaluate_n": 12, "F1": value})
#     predictions[name] = prediction
#     if name == "Forest":
#         importance = pd.DataFrame({"feature": model["pre"].get_feature_names_out(),
#                                    "importance": model["model"].feature_importances_})
# comparison = pd.DataFrame(rows)
# comparison.to_csv(OUT / "classification_comparison.csv", index=False, encoding="utf-8-sig")
# pd.DataFrame(cv_rows).to_csv(OUT / "cv_scores.csv", index=False, encoding="utf-8-sig")
# predictions.to_csv(OUT / "validation_predictions.csv", index=False, encoding="utf-8-sig")
# importance.to_csv(OUT / "feature_importance.csv", index=False, encoding="utf-8-sig")
# print(comparison.round(4).to_string(index=False))
# print("CV 평균 F1 최대, 동률이면 미리 제시한 후보 순서로 선택합니다. valid는 별도 진단입니다.")
# print("저장: outputs/classification_comparison.csv, cv_scores.csv, validation_predictions.csv, feature_importance.csv")
# print("test는 학습·변환·예측·평가하지 않았습니다.")





"""실습 07~08 응용: 얕은 폴더 구조에서 분류 실험표를 완성합니다."""
from pathlib import Path
import pandas as pd

# [경로 설정] 스크립트 파일이 위치한 현재 디렉터리의 절대 경로를 자동으로 탐색합니다.
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
    # SimpleImputer(strategy="median"): 결측치(빈칸)를 데이터의 '중간값'으로 채웁니다. (앞의 2개 수치형 변수에 적용)
    ("num", SimpleImputer(strategy="median"), FEATURES[:2]),
    # OneHotEncoder(handle_unknown="ignore", sparse_output=False):
    #   - handle_unknown="ignore": 학습 때 없던 생소한 문자가 검증 데이터에 나타나도 에러 없이 0으로 변환 처리합니다.
    #   - sparse_output=False: 눈으로 확인하기 쉬운 일반 2차원 밀집 배열(Dense Array) 형태로 결과를 반환합니다.
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), # 뒤의 2개 범주형 변수의 결측치는 '최빈값'으로 대체
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), FEATURES[2:]),
])
# 트리는 이 실습에서 표준화하지 않습니다. 각 후보·fold마다 새 전처리기를 학습합니다.

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, confusion_matrix

# [교차검증 분할기 인덱스 생성]
# n_splits=5: 데이터를 5개로 등분하여 5-Fold 교차 검증을 수행합니다.
# shuffle=True / random_state=42: 쪼개기 전 무작위로 데이터를 섞는 규칙을 고정하여 실험 재현성을 유지합니다.
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train))

# [모델 후보군 파라미터 정의]
candidates = [
    # DummyClassifier: 기준선(Baseline) 비교용 최빈값 예측 모델 (전부 가장 빈번한 클래스로 찍음)
    ("Dummy", DummyClassifier(strategy="most_frequent")),
    
    # DecisionTreeClassifier: 단일 의사결정나무 모델
    #   - max_depth=3: 질문 트리의 최대 깊이를 제한하여 과적합을 예방합니다.
    #   - min_samples_leaf=3: 말단 리프 노드가 되기 위한 최소 데이터 수 조건입니다.
    ("Tree", DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=42)), 
    
    # RandomForestClassifier: 다수 나무 병렬 결합 앙상블 모델
    #   - n_estimators=100: 생성하여 합칠 의사결정나무의 총 개수입니다.
    #   - max_features="sqrt": 전체 피처 중 루트(제곱근) 개수만큼만 무작위 선택하여 각 나무 학습에 사용합니다.
    #   - n_jobs=1: 연산 처리에 사용할 CPU 코어 개수를 1개로 한정합니다.
    ("Forest", RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=2,
                                    max_features="sqrt", random_state=42, n_jobs=1)), 
                                    
    # XGBClassifier: 순차 오차 보정형 그라디언트 부스팅 모델
    #   - learning_rate=0.05: 이전 나무의 오차를 얼마나 보정해 나갈지 결정하는 학습률입니다.
    #   - subsample=0.8 / colsample_bytree=1.0: 각 나무 학습 시 무작위 추출할 행 데이터 비율(80%)과 열 피처 비율(100%)입니다.
    #   - reg_lambda=1.0: L2 정규화 규제 강도를 설정하여 가중치의 과도한 팽창을 억제합니다.
    #   - objective="binary:logistic": 이진 분류에 특화된 예측 확률 연산 목적함수입니다.
    #   - eval_metric="logloss": 내부 평가지표로 로그 손실 점수를 사용하여 모니터링합니다.
    #   - tree_method="hist": 데이터 연속 변수를 히스토그램으로 구간화하여 속도를 엄청나게 향상시킵니다.
    ("XGB", XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0,
                  objective="binary:logistic", eval_metric="logloss",
                  tree_method="hist", device="cpu", random_state=42, n_jobs=1))
]

rows, cv_rows = [], []
# 검증 데이터셋의 고유 인덱스와 실제 실패 여부(actual) 정답을 넘파이 배열 형태로 미리 보관해 둡니다.
predictions = pd.DataFrame({"request_id": X_valid.index, "actual": y_valid.to_numpy()})

# [비교 실험 루프] 정의된 4가지 후보군 파라미터 모델을 순차적으로 꺼내어 학습을 제어합니다.
for name, estimator in candidates:
    # 매 반복 회차마다 전처리기가 오염되지 않도록 완벽하게 복사(clone)하여 파이프라인으로 연결합니다.
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    
    # [TODO A07a] 전처리를 안전하게 가둔 전체 파이프라인에 대한 5-Fold 교차 검증을 자동 수행하고, 이후 전체 train으로 학습합니다.
    # scoring="f1": 불균형 데이터 분류를 평가하기 위해 교차검증의 타겟 스코어를 F1으로 강제 지정합니다.
    scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="f1")
    model.fit(X_train, y_train)
    # TODO 끝
    
    # [TODO A07b] 전체 train 데이터로 완벽하게 학습 완료된 파이프라인을 기동하여 검증 데이터셋(X_valid)을 최종 추론합니다.
    prediction = model.predict(X_valid)
    # TODO 끝
    
    # confusion_matrix: 실제 정답과 예측값을 매핑하여 4가지 오차 행렬 경우의 수 건수를 구합니다.
    # .ravel(): 2차원 매트릭스 결과를 [TN, FP, FN, TP] 형태의 1차원 평탄한 리스트 형태로 쫙 펼쳐 분할해 줍니다.
    tn, fp, fn, tp = confusion_matrix(y_valid, prediction, labels=[0, 1]).ravel()
    
    # zero_division=0: 정밀도/재현율 공식 계산 중 분모가 0이 되어 뻗는 연산 오류 발생 시 에러 대신 0점으로 예외 우회 처리합니다.
    rows.append({"model": name, "CV_F1": scores.mean(), "CV_std": scores.std(),
                 "train_F1": f1_score(y_train, model.predict(X_train), zero_division=0),
                 "valid_F1": f1_score(y_valid, prediction, zero_division=0), "FP": fp, "FN": fn})
    
    # 수집된 5번의 스코어를 반복문을 돌며 개별 기록용 csv 로우 리스트에 차곡차곡 누적합니다.
    for number, value in enumerate(scores, start=1):
        cv_rows.append({"model": name, "fold": number, "fit_n": 48, "evaluate_n": 12, "F1": value})
    
    # 해당 모델명으로 열(컬럼)을 새로 파서 검증 데이터셋의 정답 클래스 예측 결과물 리스트를 삽입 보관합니다.
    predictions[name] = prediction
    
    # 랜덤포레스트 학습 완료 상태인 경우, 파이프라인 내부 트리 모델을 강제 추출하여 변수별 영향도(중요도)를 파악합니다.
    if name == "Forest":
        importance = pd.DataFrame({"feature": model["pre"].get_feature_names_out(),
                                   "importance": model["model"].feature_importances_})

# [결과 통합 및 내보내기] 수집된 실험 지표 결과들을 개별 파일로 안전하게 격리 저장합니다. (한글 인코딩 반영)
comparison = pd.DataFrame(rows)
comparison.to_csv(OUT / "classification_comparison.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(cv_rows).to_csv(OUT / "cv_scores.csv", index=False, encoding="utf-8-sig")
predictions.to_csv(OUT / "validation_predictions.csv", index=False, encoding="utf-8-sig")
importance.to_csv(OUT / "feature_importance.csv", index=False, encoding="utf-8-sig")

# [최종 요약 콘솔 보고] 소수점 4자리 기준 표 형태 출력 및 실험 지침 안내 문구를 띄워줍니다.
print(comparison.round(4).to_string(index=False))
print("CV 평균 F1 최대, 동률이면 미리 제시한 후보 순서로 선택합니다. valid는 별도 진단입니다.")
print("저장: outputs/classification_comparison.csv, cv_scores.csv, validation_predictions.csv, feature_importance.csv")
print("test는 학습·변환·예측·평가하지 않았습니다.")
