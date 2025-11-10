from __future__ import annotations
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

def simple_classifier(df: pd.DataFrame, report_path: str = "reports/summary.md") -> str:
    """
    Train a basic classifier to predict target_asd from numeric/categorical features.
    Writes ROC AUC and F1 to the report.
    """
    if "target_asd" not in df.columns:
        raise ValueError("target_asd column missing.")

    y = df["target_asd"].astype(int)
    X = df.drop(columns=["target_asd","subject_id"], errors="ignore")

    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    clf = Pipeline([("pre", pre),
                    ("logreg", LogisticRegression(max_iter=1000))])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]

    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    existing = Path(report_path).read_text() if Path(report_path).exists() else ""
    block = f"\n# ML: Logistic Regression\nAUC: {auc:.3f}\n\nF1: {f1:.3f}\n"
    Path(report_path).write_text(existing + block)
    return report_path
