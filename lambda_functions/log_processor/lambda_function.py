"""
Lambda function to process logs from Kinesis and prepare for anomaly detection
"""
import json
import base64
import boto3
import re
from datetime import datetime

# Initialize clients
kinesis = boto3.client('kinesis')
s3 = boto3.client('s3')

# Configuration
VOCAB = {}  # Load from S3 in production
SEQUENCE_LENGTH = 10
OUTPUT_STREAM = 'processed-log-stream'


def parse_log_event(log_message):
    """Parse log event to extract event type"""
    # Remove timestamps and IPs
    log_message = re.sub(r'\d{6}\s+\d{6}', '', log_message)
    log_message = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', 'IP', log_message)
    log_message = re.sub(r'blk_-?\d+', 'BLK', log_message)

    # Extract event type
    words = log_message.split()
    if len(words) > 0:
        return words[0]
    return 'UNKNOWN'


def encode_event(event, vocab):
    """Encode event to vocabulary ID"""
    return vocab.get(event, vocab.get('<UNK>', 0))


def lambda_handler(event, context):
    """
    Process Kinesis stream records

    Event structure:
    {
        'Records': [
            {
                'kinesis': {
                    'data': base64_encoded_log_message
                }
            }
        ]
    }
    """

    print(f"Processing {len(event['Records'])} records")

    processed_events = []

    for record in event['Records']:
        # Decode log message
        encoded_data = record['kinesis']['data']
        log_message = base64.b64decode(encoded_data).decode('utf-8')

        print(f"Processing log: {log_message[:100]}")

        # Parse log event
        event_type = parse_log_event(log_message)
        event_id = encode_event(event_type, VOCAB)

        processed_events.append({
            'event_type': event_type,
            'event_id': event_id,
            'timestamp': datetime.now().isoformat(),
            'original_message': log_message
        })

    # When we have enough events, create sequences
    # In production, maintain a sliding window in DynamoDB

    # Forward to anomaly detector Lambda when sequence is ready
    if len(processed_events) >= SEQUENCE_LENGTH:
        # Create sequence
        sequence = [e['event_id'] for e in processed_events[:SEQUENCE_LENGTH]]

        # Invoke anomaly detector Lambda
        lambda_client = boto3.client('lambda')

        payload = {
            'sequence': sequence,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'source': 'kinesis'
            }
        }

        response = lambda_client.invoke(
            FunctionName='log-anomaly-detector',
            InvocationType='Event',  # Async
            Payload=json.dumps(payload)
        )

        print(f"Invoked anomaly detector: {response['StatusCode']}")

    return {
        'statusCode': 200,
        'body': json.dumps({
            'processed': len(processed_events),
            'message': 'Successfully processed logs'
        })
    }
