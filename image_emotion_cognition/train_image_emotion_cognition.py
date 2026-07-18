import copy
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from image_emotion_common import DATA_ROOT, MODEL_ROOT, build_training_table, find_image_session_dirs, write_json


DATA_ROOT_PATH = DATA_ROOT
MODEL_OUTPUT_DIR = MODEL_ROOT
WINDOW_SECONDS = 2.5
STEP_SECONDS = 0.5
IMAGE_ONSET_SKIP_SECONDS = 0.50
MIN_VALID_FRACTION = 0.98
MIN_LABEL_FRACTION = 0.95
TENSOR_SAMPLE_COUNT = 320
RANDOM_STATE = 20260620
CV_SPLITS = 5
MIN_TRAINING_WINDOWS = 80
MIN_CLASS_WINDOWS = 12
MAX_SELECTED_FEATURES = 160
TRAIN_FEATURE_ENSEMBLE = True
TRAIN_DEEP_MODEL = True
MIN_DEEP_WINDOWS = 120
DEEP_EPOCHS = 90
DEEP_BATCH_SIZE = 32
DEEP_PATIENCE = 14
DEEP_LEARNING_RATE = 0.001
DEEP_WEIGHT_DECAY = 0.01
DEEP_DROPOUT = 0.45
DEEP_CHANNELS_1 = 64
DEEP_CHANNELS_2 = 128
PRIMARY_VALIDATION_MODE = "session_if_possible_else_image"


class EmotionEEGConvNet(nn.Module):
    def __init__(self, channel_count, class_count):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channel_count, DEEP_CHANNELS_1, kernel_size=25, padding=12, bias=False),
            nn.BatchNorm1d(DEEP_CHANNELS_1),
            nn.ELU(),
            nn.MaxPool1d(4),
            nn.Dropout(DEEP_DROPOUT),
            nn.Conv1d(DEEP_CHANNELS_1, DEEP_CHANNELS_2, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(DEEP_CHANNELS_2),
            nn.ELU(),
            nn.MaxPool1d(4),
            nn.Dropout(DEEP_DROPOUT),
            nn.Conv1d(DEEP_CHANNELS_2, DEEP_CHANNELS_2, kernel_size=9, padding=4, groups=8, bias=False),
            nn.BatchNorm1d(DEEP_CHANNELS_2),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(DEEP_CHANNELS_2, class_count)

    def forward(self, x):
        features = self.network(x).squeeze(-1)
        return self.classifier(features)


def choose_validation_groups(window_metadata, fallback_groups):
    if len(window_metadata) == 0:
        return fallback_groups, "fallback_group"
    session_ids = window_metadata["session_id"].astype(str).to_numpy()
    image_ids = window_metadata["image_id"].astype(str).to_numpy()
    repeated = window_metadata.groupby("image_id")["session_id"].nunique()
    repeated_across_sessions = bool((repeated > 1).any())
    if PRIMARY_VALIDATION_MODE == "session_if_possible_else_image" and len(np.unique(session_ids)) >= 2 and not repeated_across_sessions:
        return session_ids, "session"
    if len(np.unique(image_ids)) >= 2:
        return image_ids, "image_id"
    return fallback_groups, "trial"


def make_cv_splits(labels, groups):
    labels = np.asarray(labels, dtype=object)
    groups = np.asarray(groups, dtype=object)
    unique_groups = np.unique(groups)
    class_counts = pd.Series(labels).value_counts()
    if len(unique_groups) < 2 or len(class_counts) < 2:
        return []
    split_count = min(CV_SPLITS, len(unique_groups), int(class_counts.min()))
    if split_count < 2:
        return []
    splitter = StratifiedGroupKFold(n_splits=split_count, shuffle=True, random_state=RANDOM_STATE)
    raw_splits = list(splitter.split(np.zeros(len(labels)), labels, groups))
    classes = set(np.unique(labels).tolist())
    splits = []
    for train_indices, valid_indices in raw_splits:
        train_classes = set(np.unique(labels[train_indices]).tolist())
        valid_classes = set(np.unique(labels[valid_indices]).tolist())
        if train_classes == classes and len(valid_classes) >= 2:
            splits.append((train_indices, valid_indices))
    return splits


def metric_summary(true_labels, predicted_labels, class_names):
    return {
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(f1_score(true_labels, predicted_labels, average="macro", zero_division=0)),
        "classification_report": classification_report(true_labels, predicted_labels, labels=class_names, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels, labels=class_names).tolist(),
        "classes": list(class_names),
    }


def make_feature_estimators(feature_count, random_state):
    selected_count = min(MAX_SELECTED_FEATURES, int(feature_count))
    if selected_count >= feature_count:
        selected_count = "all"
    linear = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SelectKBest(mutual_info_classif, k=selected_count),
        LogisticRegression(max_iter=5000, C=0.18, class_weight="balanced", solver="lbfgs"),
    )
    svc = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SelectKBest(mutual_info_classif, k=selected_count),
        SVC(C=0.8, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=random_state),
    )
    trees = make_pipeline(
        SimpleImputer(strategy="median"),
        ExtraTreesClassifier(
            n_estimators=700,
            max_features="sqrt",
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    )
    hist = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            learning_rate=0.035,
            max_iter=350,
            max_leaf_nodes=15,
            l2_regularization=0.85,
            validation_fraction=0.15,
            n_iter_no_change=24,
            random_state=random_state,
        ),
    )
    return [linear, svc, trees, hist]


def model_classes(model):
    if hasattr(model, "classes_"):
        return list(model.classes_)
    return list(model[-1].classes_)


def probabilities_from_model(model, features, class_names):
    raw = model.predict_proba(features)
    output = np.zeros((len(features), len(class_names)), dtype=float)
    classes = model_classes(model)
    for class_index, class_name in enumerate(class_names):
        if class_name in classes:
            output[:, class_index] = raw[:, classes.index(class_name)]
    return output


def fit_feature_ensemble(features, labels, random_state):
    estimators = []
    for estimator in make_feature_estimators(features.shape[1], random_state):
        model = clone(estimator)
        model.fit(features, labels)
        estimators.append(model)
    return estimators


def ensemble_probabilities(estimators, features, class_names):
    probabilities = [probabilities_from_model(estimator, features, class_names) for estimator in estimators]
    return np.mean(np.stack(probabilities, axis=0), axis=0)


def cross_validate_feature_ensemble(feature_frame, labels, validation_groups, class_names, splits):
    features = feature_frame.to_numpy(dtype=float)
    oof_probabilities = np.full((len(labels), len(class_names)), np.nan, dtype=float)
    fold_summaries = []
    for fold_index, (train_indices, valid_indices) in enumerate(splits):
        estimators = fit_feature_ensemble(features[train_indices], labels[train_indices], RANDOM_STATE + fold_index * 41)
        probabilities = ensemble_probabilities(estimators, features[valid_indices], class_names)
        oof_probabilities[valid_indices] = probabilities
        predictions = np.asarray(class_names, dtype=object)[np.argmax(probabilities, axis=1)]
        fold_summaries.append({
            "fold": int(fold_index + 1),
            "train_windows": int(len(train_indices)),
            "valid_windows": int(len(valid_indices)),
            "train_groups": int(len(np.unique(validation_groups[train_indices]))),
            "valid_groups": int(len(np.unique(validation_groups[valid_indices]))),
            "macro_f1": float(f1_score(labels[valid_indices], predictions, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(labels[valid_indices], predictions)),
        })
    metrics = {"available": bool(np.all(np.isfinite(oof_probabilities))), "folds": fold_summaries}
    if np.all(np.isfinite(oof_probabilities)):
        predictions = np.asarray(class_names, dtype=object)[np.argmax(oof_probabilities, axis=1)]
        metrics.update(metric_summary(labels, predictions, class_names))
    final_estimators = fit_feature_ensemble(features, labels, RANDOM_STATE + 1009)
    return final_estimators, oof_probabilities, metrics


def tensor_normalization(train_x):
    mean = train_x.mean(axis=(0, 2), keepdims=True)
    std = train_x.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_tensors(x, mean, std):
    return ((x - mean) / std).astype(np.float32)


def predict_deep_probabilities(model, x, device):
    model.eval()
    outputs = []
    loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)), batch_size=DEEP_BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for (batch_x,) in loader:
            logits = model(batch_x.to(device))
            probabilities = torch.softmax(logits, dim=1)
            outputs.append(probabilities.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def train_deep_fold(x_train, y_train, x_valid, y_valid, class_count, random_state):
    torch.manual_seed(int(random_state))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = tensor_normalization(x_train)
    train_x = normalize_tensors(x_train, mean, std)
    valid_x = normalize_tensors(x_valid, mean, std)
    train_dataset = TensorDataset(torch.tensor(train_x, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=DEEP_BATCH_SIZE, shuffle=True, drop_last=False)
    model = EmotionEEGConvNet(train_x.shape[1], class_count).to(device)
    classes = np.arange(class_count)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device), label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEEP_LEARNING_RATE, weight_decay=DEEP_WEIGHT_DECAY)
    best_score = -1.0
    best_loss = np.inf
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = DEEP_PATIENCE
    for epoch in range(1, DEEP_EPOCHS + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x.to(device))
            loss = loss_fn(logits, batch_y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
        valid_probabilities = predict_deep_probabilities(model, valid_x, device)
        valid_predictions = np.argmax(valid_probabilities, axis=1)
        valid_loss_values = -np.log(valid_probabilities[np.arange(len(y_valid)), y_valid] + 1e-12)
        valid_loss = float(np.mean(valid_loss_values))
        score = float(f1_score(y_valid, valid_predictions, average="macro", zero_division=0))
        improved = score > best_score + 1e-4 or (abs(score - best_score) <= 1e-4 and valid_loss < best_loss)
        if improved:
            best_score = score
            best_loss = valid_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = DEEP_PATIENCE
        else:
            patience_left -= 1
        if patience_left <= 0:
            break
    model.load_state_dict(best_state)
    valid_probabilities = predict_deep_probabilities(model, valid_x, device)
    return model, valid_probabilities, {
        "best_epoch": int(best_epoch),
        "best_macro_f1": float(best_score),
        "best_loss": float(best_loss),
        "device": str(device),
    }, mean, std


def cross_validate_deep_model(tensors, label_indices, validation_groups, class_names, splits):
    oof_probabilities = np.full((len(label_indices), len(class_names)), np.nan, dtype=float)
    fold_summaries = []
    if len(tensors) < MIN_DEEP_WINDOWS:
        return None, None, oof_probabilities, {"available": False, "reason": f"Need at least {MIN_DEEP_WINDOWS} windows for deep training.", "folds": []}
    for fold_index, (train_indices, valid_indices) in enumerate(splits):
        model, probabilities, summary, mean, std = train_deep_fold(
            tensors[train_indices],
            label_indices[train_indices],
            tensors[valid_indices],
            label_indices[valid_indices],
            len(class_names),
            RANDOM_STATE + fold_index * 137,
        )
        oof_probabilities[valid_indices] = probabilities
        predictions = np.argmax(probabilities, axis=1)
        summary.update({
            "fold": int(fold_index + 1),
            "train_windows": int(len(train_indices)),
            "valid_windows": int(len(valid_indices)),
            "train_groups": int(len(np.unique(validation_groups[train_indices]))),
            "valid_groups": int(len(np.unique(validation_groups[valid_indices]))),
            "balanced_accuracy": float(balanced_accuracy_score(label_indices[valid_indices], predictions)),
        })
        fold_summaries.append(summary)
    metrics = {"available": bool(np.all(np.isfinite(oof_probabilities))), "folds": fold_summaries}
    if np.all(np.isfinite(oof_probabilities)):
        predicted_indices = np.argmax(oof_probabilities, axis=1)
        predicted_labels = np.asarray(class_names, dtype=object)[predicted_indices]
        true_labels = np.asarray(class_names, dtype=object)[label_indices]
        metrics.update(metric_summary(true_labels, predicted_labels, class_names))
    best_epochs = [fold["best_epoch"] for fold in fold_summaries if fold.get("best_epoch", 0) > 0]
    final_epochs = int(np.median(best_epochs)) if len(best_epochs) > 0 else min(DEEP_EPOCHS, 30)
    final_model, final_mean, final_std = train_deep_final(tensors, label_indices, len(class_names), final_epochs)
    return final_model, {"mean": final_mean, "std": final_std, "epochs": final_epochs}, oof_probabilities, metrics


def train_deep_final(tensors, label_indices, class_count, epochs):
    torch.manual_seed(RANDOM_STATE + 911)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = tensor_normalization(tensors)
    train_x = normalize_tensors(tensors, mean, std)
    dataset = TensorDataset(torch.tensor(train_x, dtype=torch.float32), torch.tensor(label_indices, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=DEEP_BATCH_SIZE, shuffle=True, drop_last=False)
    model = EmotionEEGConvNet(train_x.shape[1], class_count).to(device)
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(class_count), y=label_indices)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device), label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEEP_LEARNING_RATE, weight_decay=DEEP_WEIGHT_DECAY)
    for epoch in range(1, max(1, int(epochs)) + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x.to(device))
            loss = loss_fn(logits, batch_y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
    return model.cpu(), mean.astype(np.float32), std.astype(np.float32)


def class_balance(labels):
    counts = pd.Series(labels).value_counts().sort_index()
    return {str(index): int(value) for index, value in counts.items()}


def leakage_audit(window_metadata, validation_group_level):
    image_session_counts = window_metadata.groupby("image_id")["session_id"].nunique() if len(window_metadata) > 0 else pd.Series(dtype=int)
    group_sizes = window_metadata.groupby("image_id")["window_start_sample"].count() if len(window_metadata) > 0 else pd.Series(dtype=int)
    return {
        "validation_group_level": validation_group_level,
        "window_count": int(len(window_metadata)),
        "session_count": int(window_metadata["session_id"].nunique()) if len(window_metadata) > 0 else 0,
        "image_count": int(window_metadata["image_id"].nunique()) if len(window_metadata) > 0 else 0,
        "images_repeated_across_sessions": int((image_session_counts > 1).sum()) if len(image_session_counts) > 0 else 0,
        "max_windows_per_image": int(group_sizes.max()) if len(group_sizes) > 0 else 0,
        "image_metadata_used_as_features": False,
        "image_pixels_used_as_features": False,
        "validation_note": "All overlapping windows from a held-out group are kept together. The model sees only EEG-derived tensors/features plus the training label.",
    }


def save_outputs(feature_bundle, deep_bundle, metrics, training_windows):
    output_dir = Path(MODEL_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    metrics_path = output_dir / f"image_emotion_metrics_{stamp}.json"
    windows_path = output_dir / f"image_emotion_training_windows_{stamp}.csv"
    feature_model_path = output_dir / f"image_emotion_feature_model_{stamp}.pkl"
    deep_model_path = output_dir / f"image_emotion_deep_model_{stamp}.pt"
    training_windows.to_csv(windows_path, index=False)
    write_json(metrics_path, metrics)
    if feature_bundle is not None:
        with open(feature_model_path, "wb") as file:
            pickle.dump(feature_bundle, file)
        (output_dir / "latest_feature_model_path.txt").write_text(str(feature_model_path), encoding="utf-8")
    else:
        feature_model_path = None
    if deep_bundle is not None:
        torch.save(deep_bundle, deep_model_path)
        (output_dir / "latest_deep_model_path.txt").write_text(str(deep_model_path), encoding="utf-8")
    else:
        deep_model_path = None
    (output_dir / "latest_metrics_path.txt").write_text(str(metrics_path), encoding="utf-8")
    return feature_model_path, deep_model_path, metrics_path, windows_path


def main():
    session_dirs = find_image_session_dirs(DATA_ROOT_PATH)
    if len(session_dirs) == 0:
        raise FileNotFoundError(f"No image emotion sessions found under {DATA_ROOT_PATH}")
    feature_frame, labels, trial_groups, window_metadata, tensors, channel_names = build_training_table(
        session_dirs,
        WINDOW_SECONDS,
        STEP_SECONDS,
        IMAGE_ONSET_SKIP_SECONDS,
        MIN_VALID_FRACTION,
        MIN_LABEL_FRACTION,
        TENSOR_SAMPLE_COUNT,
    )
    if len(feature_frame) < MIN_TRAINING_WINDOWS:
        raise RuntimeError(f"Only {len(feature_frame)} usable EEG windows found. Collect more sessions before training.")
    counts = pd.Series(labels).value_counts()
    if len(counts) < 2:
        raise RuntimeError("Need at least two emotion classes before training.")
    if int(counts.min()) < MIN_CLASS_WINDOWS:
        raise RuntimeError(f"Smallest class has only {int(counts.min())} windows. Need at least {MIN_CLASS_WINDOWS}.")
    validation_groups, validation_group_level = choose_validation_groups(window_metadata, trial_groups)
    splits = make_cv_splits(labels, validation_groups)
    if len(splits) == 0:
        raise RuntimeError("Could not create grouped cross-validation folds. Collect more images/classes before training.")
    label_encoder = LabelEncoder()
    label_indices = label_encoder.fit_transform(labels)
    class_names = list(label_encoder.classes_)
    metrics = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "session_dirs": [str(path) for path in session_dirs],
        "window_count": int(len(feature_frame)),
        "feature_count": int(feature_frame.shape[1]),
        "tensor_shape": [int(value) for value in tensors.shape],
        "channel_names": channel_names,
        "class_balance": class_balance(labels),
        "validation_group_level": validation_group_level,
        "validation_group_count": int(len(np.unique(validation_groups))),
        "cv_splits": int(len(splits)),
        "leakage_audit": leakage_audit(window_metadata, validation_group_level),
        "training_settings": {
            "window_seconds": WINDOW_SECONDS,
            "step_seconds": STEP_SECONDS,
            "image_onset_skip_seconds": IMAGE_ONSET_SKIP_SECONDS,
            "min_valid_fraction": MIN_VALID_FRACTION,
            "min_label_fraction": MIN_LABEL_FRACTION,
            "tensor_sample_count": TENSOR_SAMPLE_COUNT,
            "random_state": RANDOM_STATE,
            "primary_validation_mode": PRIMARY_VALIDATION_MODE,
        },
    }
    feature_bundle = None
    feature_oof_probabilities = np.full((len(labels), len(class_names)), np.nan, dtype=float)
    if TRAIN_FEATURE_ENSEMBLE:
        feature_estimators, feature_oof_probabilities, feature_metrics = cross_validate_feature_ensemble(
            feature_frame,
            labels,
            validation_groups,
            class_names,
            splits,
        )
        metrics["feature_ensemble"] = feature_metrics
        feature_bundle = {
            "created_at": metrics["created_at"],
            "model_type": "eeg_only_feature_ensemble",
            "feature_names": list(feature_frame.columns),
            "class_names": class_names,
            "estimators": feature_estimators,
            "channel_names": channel_names,
            "settings": metrics["training_settings"],
            "metrics": feature_metrics,
        }
    deep_bundle = None
    deep_oof_probabilities = np.full((len(labels), len(class_names)), np.nan, dtype=float)
    if TRAIN_DEEP_MODEL:
        deep_model, deep_normalization, deep_oof_probabilities, deep_metrics = cross_validate_deep_model(
            tensors,
            label_indices,
            validation_groups,
            class_names,
            splits,
        )
        metrics["deep_convnet"] = deep_metrics
        if deep_model is not None:
            deep_bundle = {
                "created_at": metrics["created_at"],
                "model_type": "eeg_only_temporal_convnet",
                "state_dict": deep_model.state_dict(),
                "class_names": class_names,
                "channel_names": channel_names,
                "tensor_sample_count": TENSOR_SAMPLE_COUNT,
                "normalization": deep_normalization,
                "settings": metrics["training_settings"],
                "metrics": deep_metrics,
                "architecture": {
                    "channels_1": DEEP_CHANNELS_1,
                    "channels_2": DEEP_CHANNELS_2,
                    "dropout": DEEP_DROPOUT,
                    "weight_decay": DEEP_WEIGHT_DECAY,
                },
            }
    training_windows = window_metadata.copy()
    training_windows["label"] = labels
    training_windows["validation_group"] = validation_groups
    for class_index, class_name in enumerate(class_names):
        training_windows[f"feature_oof_probability_{class_name}"] = feature_oof_probabilities[:, class_index]
        training_windows[f"deep_oof_probability_{class_name}"] = deep_oof_probabilities[:, class_index]
    if np.all(np.isfinite(feature_oof_probabilities)):
        training_windows["feature_oof_prediction"] = np.asarray(class_names, dtype=object)[np.argmax(feature_oof_probabilities, axis=1)]
    else:
        training_windows["feature_oof_prediction"] = "unavailable"
    if np.all(np.isfinite(deep_oof_probabilities)):
        training_windows["deep_oof_prediction"] = np.asarray(class_names, dtype=object)[np.argmax(deep_oof_probabilities, axis=1)]
    else:
        training_windows["deep_oof_prediction"] = "unavailable"
    feature_model_path, deep_model_path, metrics_path, windows_path = save_outputs(feature_bundle, deep_bundle, metrics, training_windows)
    print("Sessions:")
    for session_dir in session_dirs:
        print(session_dir)
    print("Saved feature model:", feature_model_path)
    print("Saved deep model:", deep_model_path)
    print("Saved metrics:", metrics_path)
    print("Saved training windows:", windows_path)
    print(metrics)


if __name__ == "__main__":
    main()
