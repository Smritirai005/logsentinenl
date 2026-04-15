"""
Data drift detection for log anomaly detection
"""
import numpy as np
import boto3
import yaml
from scipy.stats import ks_2samp, entropy
from datetime import datetime, timedelta
from loguru import logger
import json


class LogDriftDetector:
    """Detects drift in log event distributions"""

    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.cloudwatch = boto3.client('cloudwatch')
        self.sns = boto3.client('sns')

        # Load reference distribution (from training data)
        self.reference_dist = None
        self.load_reference_distribution()

    def load_reference_distribution(self):
        """Load reference distribution from training data"""
        logger.info("Loading reference distribution...")

        try:
            train_seq = np.load("data/processed/train_sequences.npy")

            # Calculate event frequency distribution
            flat_events = train_seq.flatten()
            unique, counts = np.unique(flat_events, return_counts=True)

            # Normalize to probability distribution
            self.reference_dist = counts / counts.sum()

            logger.info(f"Loaded reference distribution with {len(unique)} unique events")

        except FileNotFoundError:
            logger.warning("Reference distribution not found. Run preprocessing first.")
            self.reference_dist = None

    def calculate_kl_divergence(self, current_dist):
        """Calculate KL divergence between current and reference distributions"""
        if self.reference_dist is None:
            logger.error("Reference distribution not loaded!")
            return None

        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        current_dist = current_dist + epsilon
        reference_dist = self.reference_dist + epsilon

        # Ensure same length
        if len(current_dist) != len(reference_dist):
            max_len = max(len(current_dist), len(reference_dist))
            current_dist = np.pad(current_dist, (0, max_len - len(current_dist)))
            reference_dist = np.pad(reference_dist, (0, max_len - len(reference_dist)))

        # Calculate KL divergence
        kl_div = entropy(reference_dist, current_dist)

        return kl_div

    def detect_drift(self, current_sequences):
        """Detect drift in current log sequences"""
        logger.info("Checking for drift...")

        # Calculate current distribution
        flat_events = current_sequences.flatten()
        unique, counts = np.unique(flat_events, return_counts=True)
        current_dist = counts / counts.sum()

        # Calculate KL divergence
        kl_divergence = self.calculate_kl_divergence(current_dist)

        if kl_divergence is None:
            return None

        # Check threshold
        threshold = self.config['monitoring']['alert_threshold']
        drift_detected = kl_divergence > threshold

        logger.info(f"KL Divergence: {kl_divergence:.6f}")
        logger.info(f"Threshold: {threshold}")
        logger.info(f"Drift Detected: {drift_detected}")

        # Log to CloudWatch
        self.log_drift_metric(kl_divergence)

        # Send alert if drift detected
        if drift_detected:
            self.send_drift_alert(kl_divergence, threshold)

        return {
            'drift_detected': drift_detected,
            'kl_divergence': float(kl_divergence),
            'threshold': threshold,
            'timestamp': datetime.now().isoformat()
        }

    def log_drift_metric(self, kl_divergence):
        """Log drift metric to CloudWatch"""
        try:
            self.cloudwatch.put_metric_data(
                Namespace='LogAnomalyDetection',
                MetricData=[
                    {
                        'MetricName': 'KLDivergence',
                        'Value': kl_divergence,
                        'Unit': 'None',
                        'Timestamp': datetime.now()
                    }
                ]
            )
            logger.info("Logged drift metric to CloudWatch")
        except Exception as e:
            logger.error(f"Failed to log metric: {e}")

    def send_drift_alert(self, kl_divergence, threshold):
        """Send SNS alert when drift is detected"""
        try:
            topic_arn = self.config['aws']['sns']['topic_arn']

            message = f"""
DATA DRIFT DETECTED!

KL Divergence: {kl_divergence:.6f}
Threshold: {threshold}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

The distribution of log events has changed significantly.
Consider retraining the model.
"""

            self.sns.publish(
                TopicArn=topic_arn,
                Subject='[ALERT] Log Anomaly Detection - Data Drift',
                Message=message
            )

            logger.info("Drift alert sent via SNS")
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    def check_drift_from_s3(self, s3_bucket, s3_key):
        """Check drift on new log sequences from S3"""
        logger.info(f"Loading sequences from s3://{s3_bucket}/{s3_key}")

        s3 = boto3.client('s3')

        # Download sequences
        local_path = '/tmp/current_sequences.npy'
        s3.download_file(s3_bucket, s3_key, local_path)

        # Load and check drift
        current_sequences = np.load(local_path)
        result = self.detect_drift(current_sequences)

        return result


def monitor_drift():
    """Continuous drift monitoring (run as scheduled job)"""
    logger.info("Starting drift monitoring...")

    detector = LogDriftDetector()

    # In production, this would check recent logs from Kinesis/S3
    # For now, we'll use test data

    try:
        test_seq = np.load("data/processed/test_sequences.npy")
        result = detector.detect_drift(test_seq)

        logger.info(f"Drift check result: {json.dumps(result, indent=2)}")

    except Exception as e:
        logger.error(f"Drift monitoring failed: {e}")


if __name__ == "__main__":
    monitor_drift()
