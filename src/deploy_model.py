"""
Deploy trained model to SageMaker
"""
import os
import tarfile
import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlowModel
import yaml
from loguru import logger


def package_model():
    """Package model into tar.gz as SageMaker requires"""
    model_path = "models/lstm_saved_model"
    tar_path = "models/model.tar.gz"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run training first."
        )

    logger.info(f"Packaging {model_path} into {tar_path}")

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(model_path, arcname="1")  # version folder required by SageMaker

    logger.info("Model packaged successfully")
    return tar_path


def deploy_to_sagemaker():
    """Deploy LSTM Autoencoder to SageMaker endpoint"""
    logger.info("Deploying model to SageMaker...")

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    aws_config = config['aws']

    # Initialize SageMaker session
    session = sagemaker.Session()
    role = aws_config['sagemaker']['role_arn']

    # Package model into tar.gz (SageMaker requirement)
    tar_path = package_model()

    # Upload model to S3
    s3_bucket = aws_config['s3_bucket']
    s3_key = 'models/model.tar.gz'

    logger.info(f"Uploading model to s3://{s3_bucket}/{s3_key}")

    s3_client = boto3.client('s3', region_name=aws_config['region'])
    s3_client.upload_file(tar_path, s3_bucket, s3_key)

    model_data = f"s3://{s3_bucket}/{s3_key}"
    logger.info(f"Model uploaded to {model_data}")

    # Create SageMaker TensorFlow model — no custom inference code needed
    # SageMaker TF container serves the SavedModel natively via TF Serving
    tensorflow_model = TensorFlowModel(
        model_data=model_data,
        role=role,
        framework_version="2.12",
        sagemaker_session=session
        # ✅ No image_uri, no py_version, no entry_point, no source_dir
    )

    # Deploy endpoint
    endpoint_name = aws_config['sagemaker']['endpoint_name']
    instance_type = aws_config['sagemaker']['instance_type']

    logger.info(f"Creating endpoint: {endpoint_name} (this takes ~5-10 minutes...)")

    predictor = tensorflow_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name
    )

    logger.info(f"✅ Model deployed to endpoint: {endpoint_name}")
    return predictor


if __name__ == "__main__":
    deploy_to_sagemaker()