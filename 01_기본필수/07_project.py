# """실습 07~08: 분류 후보 2개를 비교하고 같은 파일을 다시 실행합니다."""
# from pathlib import Path
# import pandas as pd
# from sklearn.model_selection import train_test_split

# # 제공: 이 파일 옆의 정상 CSV 20건을 읽습니다. 현재 터미널 위치와 무관합니다.
# HERE = Path(__file__).resolve().parent
# raw = pd.read_csv(HERE / "requests.csv")
# develop, test = train_test_split(raw, test_size=0.2, stratify=raw["failed"], random_state=42)
# train, valid = train_test_split(develop, test_size=0.25, stratify=develop["failed"], random_state=42)
# # test 4건은 보관만 합니다. 입력은 실행 전에 아는 글자 수입니다.
# X_train = train[["input_chars"]]
# X_valid = valid[["input_chars"]]
# y_train = train["failed"]
# y_valid = valid["failed"]

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import StratifiedKFold, cross_val_score
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.base import clone
# from sklearn.metrics import f1_score

# # 제공: 전날과 같은 입력 2열입니다. 실행시간과 failed는 입력에 넣지 않습니다.
# features = ["input_chars", "task_type"]
# X_train, X_valid = train[features], valid[features]
# pre = ColumnTransformer([
#     ("num", "passthrough", ["input_chars"]),
#     ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["task_type"]),
# ])
# folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
# candidates = [("Tree", DecisionTreeClassifier(max_depth=3, random_state=42)),
#               ("Forest", RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42, n_jobs=1))]
# rows, cv_rows = [], []
# predictions = valid[["request_id", "failed"]].copy()
# for name, estimator in candidates:
#     # 제공: 전처리까지 Pipeline에 포함하여 fold 학습 부분에서만 fit하게 합니다.
#     model = Pipeline([("pre", clone(pre)), ("model", estimator)])
#     # TODO B07a: 전처리까지 연결한 model을 train에서 교차검증하세요.
#     scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="f1")
#     # TODO 끝
#     # TODO B07b: 전체 train으로 학습하고 valid를 예측하세요.
#     model.fit(X_train, y_train)
#     prediction = model.predict(X_valid)    
#     # TODO 끝
#     rows.append({"model": name, "CV_F1": scores.mean(),
#                  "CV_std": scores.std(), "valid_F1": f1_score(y_valid, prediction, zero_division=0)})
#     for number, value in enumerate(scores, start=1):
#         cv_rows.append({"model": name, "fold": number, "F1": value})
#     predictions[name] = prediction
#     if name == "Forest":
#         importance = pd.DataFrame({"feature": model["pre"].get_feature_names_out(),
#                                    "importance": model["model"].feature_importances_})

# # 제공: 입력 파일을 덮어쓰지 않고 결과 4개를 같은 폴더에 만듭니다.
# comparison = pd.DataFrame(rows)
# comparison.to_csv(HERE / "classification_comparison.csv", index=False, encoding="utf-8-sig")
# pd.DataFrame(cv_rows).to_csv(HERE / "cv_scores.csv", index=False, encoding="utf-8-sig")
# predictions.to_csv(HERE / "validation_predictions.csv", index=False, encoding="utf-8-sig")
# importance.to_csv(HERE / "feature_importance.csv", index=False, encoding="utf-8-sig")
# print("분할:", len(train), len(valid), len(test))
# print(comparison.round(3).to_string(index=False))
# print("중요도")
# print(importance.round(3).to_string(index=False))
# print("저장: classification_comparison.csv, cv_scores.csv, validation_predictions.csv, feature_importance.csv")
# print("test는 학습·변환·예측·평가하지 않았습니다.")











"""실습 07~08: 분류 후보 2개를 비교하고 같은 파일을 다시 실행합니다."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# [경로 설정] 스크립트 파일이 위치한 현재 디렉터리의 절대 경로를 자동으로 찾습니다.
HERE = Path(__file__).resolve().parent

# [데이터 로드] 현재 파일과 같은 위치에 있는 requests.csv 파일을 읽어옵니다.
raw = pd.read_csv(HERE / "requests.csv")

# [데이터 분할 1단계] 전체 데이터를 모델 개발용(80%)과 최종 평가 보관용(test, 4건)으로 분할합니다.
develop, test = train_test_split(raw, test_size=0.2, stratify=raw["failed"], random_state=42)

# [데이터 분할 2단계] 개발용 데이터를 다시 실제 학습용(train, 12건)과 내부 검증용(valid, 4건)으로 나눕니다.
train, valid = train_test_split(develop, test_size=0.25, stratify=develop["failed"], random_state=42)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import f1_score

# [피처 지정 및 분리] 모델 학습에 사용할 2개의 피처(input_chars, task_type)만 추출합니다.
features = ["input_chars", "task_type"]
X_train, X_valid = train[features], valid[features]
y_train, y_valid = train["failed"], valid["failed"]

# [전처리 설계] 수치형은 그대로 통과(passthrough)시키고, 범주형은 원-핫 인코딩을 적용하는 변환기를 만듭니다.
pre = ColumnTransformer([
    ("num", "passthrough", ["input_chars"]),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["task_type"]),
])

# [교차검증 및 모델 정의] 3-Fold 분할기와 비교 대상인 의사결정나무 및 랜덤 포레스트 모델을 정의합니다.
folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
candidates = [("Tree", DecisionTreeClassifier(max_depth=3, random_state=42)),
              ("Forest", RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42, n_jobs=1))]

rows, cv_rows = [], []
# [예측값 보관함] 검증 데이터셋의 ID와 실제 실패 정답(failed) 정보를 미리 복사해 둡니다.
predictions = valid[["request_id", "failed"]].copy()

# [실험 루프] 후보 모델들을 하나씩 꺼내어 교차 검증과 학습 및 예측을 수행합니다.
for name, estimator in candidates:
    # [파이프라인 구축] 전처리기 복사본과 머신러닝 알고리즘을 파이프라인으로 연결합니다.
    model = Pipeline([("pre", clone(pre)), ("model", estimator)])
    
    # [TODO B07a] 전처리 단계를 격리 및 포함한 파이프라인 전체의 3-Fold F1 점수를 자동 계산합니다.
    scores = cross_val_score(model, X_train, y_train, cv=folds, scoring="f1")
    # TODO 끝
    
    # [TODO B07b] 12건의 전체 train 데이터로 파이프라인을 최종 학습하고, 4건의 valid 데이터를 예측합니다.
    model.fit(X_train, y_train)
    prediction = model.predict(X_valid)
    # TODO 끝
    
    # [요약 기록] 모델별 교차 검증 F1 평균/표준편차와 검증 데이터셋 대상 실제 F1 점수를 기록합니다.
    rows.append({"model": name, "CV_F1": scores.mean(),
                 "CV_std": scores.std(), "valid_F1": f1_score(y_valid, prediction, zero_division=0)})
    
    # [상세 폴드 기록] 각 폴드별로 나온 원본 점수(3개)를 리스트에 담습니다.
    for number, value in enumerate(scores, start=1):
        cv_rows.append({"model": name, "fold": number, "F1": value})
        
    # [예측 열 추가] 최종 예측 결과(0 또는 1)를 해당 모델 이름의 컬럼으로 추가 저장합니다.
    predictions[name] = prediction
    
    # [특성 중요도 추출] 랜덤 포레스트 모델일 때, 인코딩 완료된 피처 이름들과 트리 내 피처 중요도를 매핑합니다.
    if name == "Forest":
        importance = pd.DataFrame({"feature": model["pre"].get_feature_names_out(),
                                   "importance": model["model"].feature_importances_})

# [결과 파일 저장] 파일 덮어쓰기 없이 같은 폴더 내에 4개의 csv 결과 결과물을 생성합니다. (한글 깨짐 방지 적용)
comparison = pd.DataFrame(rows)
comparison.to_csv(HERE / "classification_comparison.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(cv_rows).to_csv(HERE / "cv_scores.csv", index=False, encoding="utf-8-sig")
predictions.to_csv(HERE / "validation_predictions.csv", index=False, encoding="utf-8-sig")
importance.to_csv(HERE / "feature_importance.csv", index=False, encoding="utf-8-sig")

# [콘솔 화면 출력] 학습에 사용된 최종 데이터 건수와 종합 성능 비교표, 랜덤 포레스트의 특성 중요도를 출력합니다.
print("분할:", len(train), len(valid), len(test))
print(comparison.round(3).to_string(index=False))
print("중요도")
print(importance.round(3).to_string(index=False))
print("저장: classification_comparison.csv, cv_scores.csv, validation_predictions.csv, feature_importance.csv")
print("test는 학습·변환·예측·평가하지 않았습니다.")

