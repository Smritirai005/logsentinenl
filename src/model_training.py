"""
Train LSTM Autoencoder for log anomaly detection
"""
import numpy as np
import yaml
import joblib
from tensorflow import keras
from tensorflow.keras import layers
from loguru import logger


class LSTMAutoencoder:
    """LSTM Autoencoder for log sequence anomaly detection"""

    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_config = self.config['model']
        self.model = None
        self.threshold = None

    def build_model(self, vocab_size, sequence_length):
        """Build LSTM Autoencoder architecture"""
        logger.info("Building LSTM Autoencoder...")

        embedding_dim = self.model_config['embedding_dim']
        lstm_units = self.model_config['lstm_units']

        # Encoder
        inputs = layers.Input(shape=(sequence_length,))

        # Embedding layer
        x = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=sequence_length
        )(inputs)

        # LSTM Encoder
        encoded = layers.LSTM(lstm_units)(x)

        # Repeat for decoder
        x = layers.RepeatVector(sequence_length)(encoded)

        # LSTM Decoder
        x = layers.LSTM(lstm_units, return_sequences=True)(x)

        # Output layer
        outputs = layers.TimeDistributed(
            layers.Dense(vocab_size, activation='softmax')
        )(x)

        # Create model
        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.model_config['learning_rate']
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        logger.info("Model architecture built")
        model.summary()

        self.model = model
        return model

    def train(self, train_sequences, test_sequences):
        """Train the autoencoder"""
        logger.info("Starting training...")

        # Load vocab to get vocab_size
        vocab = np.load(
            self.config['data']['processed_path'] + 'vocab.npy',
            allow_pickle=True
        ).item()
        vocab_size = len(vocab)
        sequence_length = train_sequences.shape[1]

        # Build model
        self.build_model(vocab_size, sequence_length)

        # Prepare data (X = Y for autoencoder)
        X_train = train_sequences
        y_train = np.expand_dims(train_sequences, -1)

        X_test = test_sequences
        y_test = np.expand_dims(test_sequences, -1)

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                'models/lstm_autoencoder_best.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]

        # Train
        history = self.model.fit(
            X_train, y_train,
            epochs=self.model_config['epochs'],
            batch_size=self.model_config['batch_size'],
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )

        logger.info("Training complete!")

        # Calculate reconstruction error threshold
        self.calculate_threshold(train_sequences)

        # Save model
        self.model.export('models/lstm_saved_model')
        joblib.dump(vocab, 'models/vocab.pkl')
        joblib.dump(self.threshold, 'models/threshold.pkl')

        logger.info("Model and artifacts saved")

        return history

    def calculate_threshold(self, train_sequences):
        """Calculate anomaly threshold from reconstruction error"""
        logger.info("Calculating anomaly threshold...")

        # Predict on training data
        predictions = self.model.predict(train_sequences)

        # Calculate reconstruction error
        reconstruction_errors = np.mean(
            np.square(
                np.expand_dims(train_sequences, -1) - predictions
            ),
            axis=(1, 2)
        )

        # Set threshold at percentile
        percentile = self.model_config['threshold_percentile']
        self.threshold = np.percentile(reconstruction_errors, percentile)

        logger.info(f"Anomaly threshold: {self.threshold:.6f}")
        logger.info(f"   (95th percentile of reconstruction error)")

        return self.threshold

    def detect_anomaly(self, sequence):
        """Detect if a sequence is anomalous"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Predict
        prediction = self.model.predict(np.array([sequence]))

        # Calculate reconstruction error
        error = np.mean(
            np.square(
                np.expand_dims(sequence, -1) - prediction[0]
            )
        )

        # Check if anomaly
        is_anomaly = error > self.threshold

        return {
            'is_anomaly': bool(is_anomaly),
            'reconstruction_error': float(error),
            'threshold': float(self.threshold),
            'anomaly_score': float(error / self.threshold)
        }


def train_model():
    """Main training function"""
    logger.info("Loading preprocessed data...")

    # Load data
    train_seq = np.load("data/processed/train_sequences.npy")
    test_seq = np.load("data/processed/test_sequences.npy")

    logger.info(f"Train: {train_seq.shape}, Test: {test_seq.shape}")

    # Train model
    autoencoder = LSTMAutoencoder()
    history = autoencoder.train(train_seq, test_seq)

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_predictions = autoencoder.model.predict(test_seq)

    test_errors = np.mean(
        np.square(np.expand_dims(test_seq, -1) - test_predictions),
        axis=(1, 2)
    )

    normal_count = np.sum(test_errors <= autoencoder.threshold)
    anomaly_count = np.sum(test_errors > autoencoder.threshold)

    logger.info(f"Test set results:")
    logger.info(f"  Normal: {normal_count} ({normal_count/len(test_seq)*100:.1f}%)")
    logger.info(f"  Anomalies: {anomaly_count} ({anomaly_count/len(test_seq)*100:.1f}%)")

    logger.info("All done!")


if __name__ == "__main__":
    train_model()
