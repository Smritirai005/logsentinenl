"""
Lambda function to detect anomalies using SageMaker endpoint
"""
import json
import boto3
import numpy as np
from datetime import datetime

# Initialize clients
sagemaker_runtime = boto3.client('sagemaker-runtime')
cloudwatch = boto3.client('cloudwatch')
sns = boto3.client('sns')

# Configuration
ENDPOINT_NAME = 'log-anomaly-endpoint-v10'
ANOMALY_THRESHOLD = 0.05  # Load from environment
SNS_TOPIC_ARN = 'arn:aws:sns:us-east-1:ACCOUNT_ID:log-anomaly-alerts'


# lambda_functions/anomaly_detector/lambda_function.py

def invoke_sagemaker_endpoint(sequence):
    input_data = json.dumps({
        "instances": [sequence]   # ✅ TF Serving native format
    })
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/json',
        Body=input_data
    )
    result = json.loads(response['Body'].read().decode())
    return result  # returns {"predictions": [[...]]}


def calculate_reconstruction_error(sequence, prediction):
    """Calculate reconstruction error"""
    sequence = np.array(sequence)
    prediction = np.array(prediction)

    error = np.mean(np.square(sequence - prediction))

    return error


def log_metrics(is_anomaly, error):
    """Log metrics to CloudWatch"""

    try:
        cloudwatch.put_metric_data(
            Namespace='LogAnomalyDetection',
            MetricData=[
                {
                    'MetricName': 'AnomalyDetected',
                    'Value': 1 if is_anomaly else 0,
                    'Unit': 'Count',
                    'Timestamp': datetime.now()
                },
                {
                    'MetricName': 'ReconstructionError',
                    'Value': error,
                    'Unit': 'None',
                    'Timestamp': datetime.now()
                }
            ]
        )
        print("Logged metrics to CloudWatch")

    except Exception as e:
        print(f"Failed to log metrics: {e}")


def send_anomaly_alert(sequence, error, threshold):
    """Send SNS alert for anomaly"""

    message = f"""
LOG ANOMALY DETECTED!

Reconstruction Error: {error:.6f}
Threshold: {threshold}
Anomaly Score: {error / threshold:.2f}x
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Sequence: {sequence}

Please investigate the system logs.
"""

    try:
        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject='[ALERT] Log Anomaly Detected',
            Message=message
        )
        print("Alert sent via SNS")

    except Exception as e:
        print(f"Failed to send alert: {e}")


def lambda_handler(event, context):
    """
    Detect anomalies in log sequences

    Event structure:
    {
        'sequence': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'metadata': {
            'timestamp': '2024-01-01T00:00:00',
            'source': 'kinesis'
        }
    }
    """

    print(f"Event: {json.dumps(event)}")

    sequence = event['sequence']
    metadata = event.get('metadata', {})

    # Call SageMaker endpoint
    try:
        prediction = invoke_sagemaker_endpoint(sequence)

        print(f"Prediction: {prediction}")

        # Calculate reconstruction error
        error = calculate_reconstruction_error(sequence, prediction['predictions'][0])

        # Check if anomaly
        is_anomaly = error > ANOMALY_THRESHOLD

        print(f"Reconstruction Error: {error:.6f}")
        print(f"Is Anomaly: {is_anomaly}")

        # Log metrics
        log_metrics(is_anomaly, error)

        # Send alert if anomaly
        if is_anomaly:
            send_anomaly_alert(sequence, error, ANOMALY_THRESHOLD)

        return {
            'statusCode': 200,
            'body': json.dumps({
                'is_anomaly': is_anomaly,
                'reconstruction_error': error,
                'threshold': ANOMALY_THRESHOLD,
                'anomaly_score': error / ANOMALY_THRESHOLD,
                'timestamp': datetime.now().isoformat()
            })
        }

    except Exception as e:
        print(f"Error: {e}")

        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }
