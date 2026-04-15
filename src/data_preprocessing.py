"""
Download and preprocess HDFS log data
"""
import numpy as np
import pandas as pd
import re
import os
import requests
import yaml
from collections import Counter
from sklearn.model_selection import train_test_split
from loguru import logger


class HDFSLogPreprocessor:
    """Preprocess HDFS logs for LSTM Autoencoder"""

    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.sequence_length = self.config['data']['sequence_length']
        self.vocab = {}
        self.vocab_size = 0

    def download_data(self):
        """Download HDFS log data"""
        url = self.config['data']['url']
        output_path = self.config['data']['raw_path']

        logger.info(f"Downloading HDFS logs from {url}")

        response = requests.get(url)
        with open(output_path, 'wb') as f:
            f.write(response.content)

        logger.info(f"Downloaded to {output_path}")
        return output_path

    def parse_log_line(self, line):
        """Extract log template from line"""
        # HDFS log format: blk_id timestamp ... message
        # We focus on the message pattern

        # Remove timestamps and IPs
        line = re.sub(r'\d{6}\s+\d{6}', '', line)
        line = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', 'IP', line)
        line = re.sub(r'blk_-?\d+', 'BLK', line)

        # Extract main event type
        words = line.split()
        if len(words) > 0:
            return words[0]  # Use first word as event type
        return 'UNKNOWN'

    def build_vocabulary(self, logs):
        """Build vocabulary from log events"""
        logger.info("Building vocabulary...")

        events = [self.parse_log_line(log) for log in logs]
        event_counts = Counter(events)

        # Create vocab (most common events)
        self.vocab = {event: idx+1 for idx, (event, _) in
                      enumerate(event_counts.most_common())}
        self.vocab['<PAD>'] = 0
        self.vocab['<UNK>'] = len(self.vocab)

        self.vocab_size = len(self.vocab)
        logger.info(f"Vocabulary size: {self.vocab_size}")

        return events

    def create_sequences(self, events):
        """Create sequences for LSTM"""
        logger.info(f"Creating sequences of length {self.sequence_length}")

        # Convert events to IDs
        event_ids = [self.vocab.get(e, self.vocab['<UNK>']) for e in events]

        # Create sliding window sequences
        sequences = []
        for i in range(len(event_ids) - self.sequence_length):
            seq = event_ids[i:i + self.sequence_length]
            sequences.append(seq)

        sequences = np.array(sequences)
        logger.info(f"Created {len(sequences)} sequences")

        return sequences

    def preprocess(self):
        """Main preprocessing pipeline"""
        logger.info("Starting preprocessing...")

        # Download data
        log_file = self.download_data()

        # Read logs
        with open(log_file, 'r') as f:
            logs = f.readlines()

        logger.info(f"Loaded {len(logs)} log lines")

        # Build vocabulary
        events = self.build_vocabulary(logs)

        # Create sequences
        sequences = self.create_sequences(events)

        # Split train/test
        train_size = self.config['data']['train_split']
        train_seq, test_seq = train_test_split(
            sequences,
            train_size=train_size,
            random_state=42
        )


        output_dir = self.config['data']['processed_path']
        os.makedirs(output_dir, exist_ok=True)

        np.save(os.path.join(output_dir, "train_sequences.npy"), train_seq)
        np.save(os.path.join(output_dir, "test_sequences.npy"), test_seq)
        np.save(os.path.join(output_dir, "vocab.npy"), self.vocab)

        logger.info(f"Saved preprocessed data to {output_dir}")
        logger.info(f"Train: {len(train_seq)}, Test: {len(test_seq)}")

        return train_seq, test_seq


if __name__ == "__main__":
    preprocessor = HDFSLogPreprocessor()
    train_seq, test_seq = preprocessor.preprocess()
