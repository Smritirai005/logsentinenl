"""
Train multiple models for log anomaly detection and compare results.

Models:
  1. LSTM Autoencoder  (original)
  2. GRU  Autoencoder  (lighter recurrent baseline)
  3. Isolation Forest  (classic unsupervised)
  4. One-Class SVM     (boundary-based unsupervised)
"""

import numpy as np
import yaml
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from loguru import logger


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def flatten_sequences(sequences):
    """Flatten (N, seq_len) → (N, seq_len) as float for sklearn models."""
    return sequences.astype(np.float32)


# ─────────────────────────────────────────────
# 1. LSTM Autoencoder  (original, unchanged)
# ─────────────────────────────────────────────

class LSTMAutoencoder:
    """LSTM Autoencoder for log sequence anomaly detection."""

    MODEL_NAME = "LSTM Autoencoder"

    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.model_config = self.config["model"]
        self.model = None
        self.threshold = None

    def build_model(self, vocab_size, sequence_length):
        logger.info("Building LSTM Autoencoder...")
        embedding_dim = self.model_config["embedding_dim"]
        lstm_units    = self.model_config["lstm_units"]

        inputs  = layers.Input(shape=(sequence_length,))
        x       = layers.Embedding(vocab_size, embedding_dim, input_length=sequence_length)(inputs)
        encoded = layers.LSTM(lstm_units)(x)
        x       = layers.RepeatVector(sequence_length)(encoded)
        x       = layers.LSTM(lstm_units, return_sequences=True)(x)
        outputs = layers.TimeDistributed(layers.Dense(vocab_size, activation="softmax"))(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(self.model_config["learning_rate"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()
        self.model = model
        return model

    def train(self, train_sequences, test_sequences):
        logger.info("[LSTM] Starting training...")
        vocab = np.load(
            self.config["data"]["processed_path"] + "vocab.npy", allow_pickle=True
        ).item()
        vocab_size      = len(vocab)
        sequence_length = train_sequences.shape[1]

        self.build_model(vocab_size, sequence_length)

        X_train, y_train = train_sequences, np.expand_dims(train_sequences, -1)
        X_test,  y_test  = test_sequences,  np.expand_dims(test_sequences,  -1)

        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(
                "models/lstm_autoencoder_best.h5", monitor="val_loss", save_best_only=True
            ),
        ]

        history = self.model.fit(
            X_train, y_train,
            epochs=self.model_config["epochs"],
            batch_size=self.model_config["batch_size"],
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1,
        )

        self.calculate_threshold(train_sequences)

        self.model.export("models/lstm_saved_model")
        joblib.dump(vocab, "models/vocab.pkl")
        joblib.dump(self.threshold, "models/lstm_threshold.pkl")

        logger.info("[LSTM] Training complete. Threshold={:.6f}".format(self.threshold))
        return history

    def calculate_threshold(self, train_sequences):
        preds  = self.model.predict(train_sequences)
        errors = np.mean(np.square(np.expand_dims(train_sequences, -1) - preds), axis=(1, 2))
        self.threshold = np.percentile(errors, self.model_config["threshold_percentile"])
        return self.threshold

    def reconstruction_errors(self, sequences):
        preds = self.model.predict(sequences)
        return np.mean(np.square(np.expand_dims(sequences, -1) - preds), axis=(1, 2))

    def predict(self, sequences):
        """Return binary labels: 1 = anomaly, 0 = normal."""
        return (self.reconstruction_errors(sequences) > self.threshold).astype(int)

    def anomaly_scores(self, sequences):
        """Normalised score: error / threshold (>1 means anomaly)."""
        return self.reconstruction_errors(sequences) / self.threshold


# ─────────────────────────────────────────────
# 2. GRU Autoencoder
# ─────────────────────────────────────────────

class GRUAutoencoder:
    """
    GRU-based Autoencoder — same idea as LSTM but uses GRU cells,
    which have fewer parameters and often train faster.
    """

    MODEL_NAME = "GRU Autoencoder"

    def __init__(self, config_path="config.yaml"):
        self.config       = load_config(config_path)
        self.model_config = self.config["model"]
        self.model        = None
        self.threshold    = None

    def build_model(self, vocab_size, sequence_length):
        logger.info("Building GRU Autoencoder...")
        embedding_dim = self.model_config["embedding_dim"]
        gru_units     = self.model_config["lstm_units"]   # reuse same unit count

        inputs  = layers.Input(shape=(sequence_length,))
        x       = layers.Embedding(vocab_size, embedding_dim, input_length=sequence_length)(inputs)
        encoded = layers.GRU(gru_units)(x)                          # GRU encoder
        x       = layers.RepeatVector(sequence_length)(encoded)
        x       = layers.GRU(gru_units, return_sequences=True)(x)   # GRU decoder
        outputs = layers.TimeDistributed(layers.Dense(vocab_size, activation="softmax"))(x)

        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(self.model_config["learning_rate"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()
        self.model = model
        return model

    def train(self, train_sequences, test_sequences):
        logger.info("[GRU] Starting training...")
        vocab = np.load(
            self.config["data"]["processed_path"] + "vocab.npy", allow_pickle=True
        ).item()
        vocab_size      = len(vocab)
        sequence_length = train_sequences.shape[1]

        self.build_model(vocab_size, sequence_length)

        X_train, y_train = train_sequences, np.expand_dims(train_sequences, -1)
        X_test,  y_test  = test_sequences,  np.expand_dims(test_sequences,  -1)

        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(
                "models/gru_autoencoder_best.h5", monitor="val_loss", save_best_only=True
            ),
        ]

        history = self.model.fit(
            X_train, y_train,
            epochs=self.model_config["epochs"],
            batch_size=self.model_config["batch_size"],
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1,
        )

        self.calculate_threshold(train_sequences)
        self.model.export("models/gru_saved_model")
        joblib.dump(self.threshold, "models/gru_threshold.pkl")

        logger.info("[GRU] Training complete. Threshold={:.6f}".format(self.threshold))
        return history

    def calculate_threshold(self, train_sequences):
        preds  = self.model.predict(train_sequences)
        errors = np.mean(np.square(np.expand_dims(train_sequences, -1) - preds), axis=(1, 2))
        self.threshold = np.percentile(errors, self.model_config["threshold_percentile"])
        return self.threshold

    def reconstruction_errors(self, sequences):
        preds = self.model.predict(sequences)
        return np.mean(np.square(np.expand_dims(sequences, -1) - preds), axis=(1, 2))

    def predict(self, sequences):
        return (self.reconstruction_errors(sequences) > self.threshold).astype(int)

    def anomaly_scores(self, sequences):
        return self.reconstruction_errors(sequences) / self.threshold


# ─────────────────────────────────────────────
# 3. Isolation Forest
# ─────────────────────────────────────────────

class IsolationForestDetector:
    """
    Isolation Forest — tree-based unsupervised anomaly detector.
    Works on flattened sequences (no temporal structure needed).
    Fast to train; excellent baseline for comparison.
    """

    MODEL_NAME = "Isolation Forest"

    def __init__(self, config_path="config.yaml", contamination=0.05, n_estimators=200, random_state=42):
        self.config        = load_config(config_path)
        self.contamination = contamination
        self.model         = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()

    def _prepare(self, sequences):
        return self.scaler.transform(flatten_sequences(sequences))

    def train(self, train_sequences, test_sequences=None):
        logger.info("[IsolationForest] Starting training...")
        X_train = self.scaler.fit_transform(flatten_sequences(train_sequences))
        self.model.fit(X_train)
        joblib.dump(self.model,  "models/isolation_forest.pkl")
        joblib.dump(self.scaler, "models/isolation_forest_scaler.pkl")
        logger.info("[IsolationForest] Training complete.")

    def predict(self, sequences):
        """Return binary labels: 1 = anomaly, 0 = normal."""
        raw = self.model.predict(self._prepare(sequences))   # sklearn: -1 anomaly, +1 normal
        return ((raw == -1)).astype(int)

    def anomaly_scores(self, sequences):
        """
        Returns the negative of sklearn's decision_function so that
        higher score = more anomalous (consistent with other models).
        """
        return -self.model.decision_function(self._prepare(sequences))


# ─────────────────────────────────────────────
# 4. One-Class SVM
# ─────────────────────────────────────────────

class OneClassSVMDetector:
    """
    One-Class SVM — learns a tight hypersphere around normal data.
    Works well when anomalies are rare and structurally different.
    Uses RBF kernel on flattened, scaled sequences.
    """

    MODEL_NAME = "One-Class SVM"

    def __init__(self, config_path="config.yaml", nu=0.05, kernel="rbf", gamma="scale"):
        self.config = load_config(config_path)
        self.model  = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        self.scaler = StandardScaler()

    def _prepare(self, sequences):
        return self.scaler.transform(flatten_sequences(sequences))

    def train(self, train_sequences, test_sequences=None):
        logger.info("[OneClassSVM] Starting training (may take a minute)...")
        X_train = self.scaler.fit_transform(flatten_sequences(train_sequences))
        self.model.fit(X_train)
        joblib.dump(self.model,  "models/ocsvm.pkl")
        joblib.dump(self.scaler, "models/ocsvm_scaler.pkl")
        logger.info("[OneClassSVM] Training complete.")

    def predict(self, sequences):
        raw = self.model.predict(self._prepare(sequences))
        return ((raw == -1)).astype(int)

    def anomaly_scores(self, sequences):
        return -self.model.decision_function(self._prepare(sequences))


# ─────────────────────────────────────────────
# Main: train all models & save comparison data
# ─────────────────────────────────────────────

def train_all_models():
    logger.info("=" * 60)
    logger.info("Loading preprocessed data...")
    logger.info("=" * 60)

    train_seq = np.load("data/processed/train_sequences.npy")
    test_seq  = np.load("data/processed/test_sequences.npy")
    logger.info(f"Train: {train_seq.shape}, Test: {test_seq.shape}")

    results = {}   # { model_name: { 'labels': [...], 'scores': [...] } }

    # ── 1. LSTM Autoencoder ──────────────────
    lstm = LSTMAutoencoder()
    lstm.train(train_seq, test_seq)
    results["LSTM Autoencoder"] = {
        "labels": lstm.predict(test_seq),
        "scores": lstm.anomaly_scores(test_seq),
        "errors": lstm.reconstruction_errors(test_seq),
        "threshold": lstm.threshold,
    }

    # ── 2. GRU Autoencoder ───────────────────
    gru = GRUAutoencoder()
    gru.train(train_seq, test_seq)
    results["GRU Autoencoder"] = {
        "labels": gru.predict(test_seq),
        "scores": gru.anomaly_scores(test_seq),
        "errors": gru.reconstruction_errors(test_seq),
        "threshold": gru.threshold,
    }

    # ── 3. Isolation Forest ──────────────────
    iso = IsolationForestDetector()
    iso.train(train_seq, test_seq)
    results["Isolation Forest"] = {
        "labels": iso.predict(test_seq),
        "scores": iso.anomaly_scores(test_seq),
    }

    # ── 4. One-Class SVM ─────────────────────
    ocsvm = OneClassSVMDetector()
    ocsvm.train(train_seq, test_seq)
    results["One-Class SVM"] = {
        "labels": ocsvm.predict(test_seq),
        "scores": ocsvm.anomaly_scores(test_seq),
    }

    # ── Save comparison results ───────────────
    joblib.dump(results, "models/all_model_results.pkl")
    logger.info("All model results saved to models/all_model_results.pkl")

    # ── Print summary table ───────────────────
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON SUMMARY (Test Set)")
    logger.info("=" * 60)
    logger.info(f"{'Model':<22} {'Anomalies':>10} {'Normal':>10} {'Anomaly %':>12}")
    logger.info("-" * 60)
    for name, res in results.items():
        n_anom  = int(res["labels"].sum())
        n_norm  = len(res["labels"]) - n_anom
        pct     = n_anom / len(res["labels"]) * 100
        logger.info(f"{name:<22} {n_anom:>10} {n_norm:>10} {pct:>11.1f}%")

    return results


if __name__ == "__main__":
    train_all_models()