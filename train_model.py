import pandas as pd
import joblib
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

def load_dataset(csv_file, label_column):
    df = pd.read_csv(csv_file)
    X = df.select_dtypes(include=['int64', 'float64']).copy()
    y = df[label_column]
    X = X.drop(columns=[label_column]) if label_column in X.columns else X
    return X, y

def preprocess_features(X, y):
    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=min(15, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    selected_columns = X.columns[selector.get_support()]
    print(f"[INFO] Selected features: {list(selected_columns)}")

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_selected, y)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    return X_scaled, y_resampled, scaler, selector

def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, target_names=['Normal', 'Anomaly'])
    f1 = f1_score(y_test, preds, pos_label=1)
    
    # Generate visualizations
    generate_visualizations(y_test, preds, name)
    
    print(f"\n[INFO] {name} Evaluation:\n{report}")
    return f1, report, preds

def generate_visualizations(y_true, y_pred, model_name):
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
    disp.plot(cmap="Blues")
    plt.title(f"Upload_Confusion Matrix - {model_name}")
    plt.savefig(f"upload_confusion_matrix_{model_name}.png")
    plt.close()
    
    # Metrics Bar Plot
    report = classification_report(y_true, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose().drop(["accuracy", "macro avg", "weighted avg"])
    
    df_report[["precision", "recall", "f1-score"]].plot(
        kind="bar", 
        figsize=(8, 5), 
        color=["skyblue", "orange", "green"]
    )
    plt.title(f"Evaluation Metrics by Class - {model_name}")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.ylim(0, 1.05)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"upload_metrics_{model_name}.png")
    plt.close()

def save_model(model, scaler, selector, filename="model.pkl"):
    joblib.dump((model, scaler, selector), filename)
    print(f"[INFO] Best model saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate Multiple Models for NIDS")
    parser.add_argument("dataset", help="Path to the CSV dataset")
    parser.add_argument("--label", required=True, help="Label column name (e.g., 'label')")
    parser.add_argument("--output", default="model.pkl", help="Path to save the best model")
    args = parser.parse_args()

    print("[INFO] Loading dataset...")
    X, y = load_dataset(args.dataset, args.label)

    print("[INFO] Preprocessing features and balancing data...")
    X_train_scaled, y_train, scaler, selector = preprocess_features(X, y)

    print("[INFO] Splitting into train/test sets...")
    X_train, X_test, y_train_split, y_test = train_test_split(
        X_train_scaled, 
        y_train, 
        test_size=0.2, 
        random_state=42
    )

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000)
    }

    results = {}
    best_model = None
    best_score = 0
    best_name = ""
    best_preds = None

    print("[INFO] Training and evaluating models...")
    for name, model in models.items():
        model.fit(X_train, y_train_split)
        f1, report, preds = evaluate_model(name, model, X_test, y_test)
        results[name] = report
        if f1 > best_score:
            best_score = f1
            best_model = model
            best_name = name
            best_preds = preds

    # Save report
    with open("evaluation_report.txt", "w") as f:
        f.write(f"=== Model Evaluation Summary ===\n\n")
        for name, report in results.items():
            f.write(f"\n>>> {name}\n")
            f.write(report)
        f.write(f"\n\nBest model: {best_name} (F1-score for Anomaly = {best_score:.4f})\n")

    # Save best model
    save_model(best_model, scaler, selector, args.output)