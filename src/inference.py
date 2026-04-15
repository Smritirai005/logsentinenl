import numpy as np
import tensorflow as tf
import json

def model_fn(model_dir):
    model = tf.keras.models.load_model(model_dir)
    return model

def input_fn(request_body, request_content_type):
    data = json.loads(request_body)
    # Support both 'instances' (Lambda) and 'inputs' key
    inputs = data.get("instances", data.get("inputs"))
    return np.array(inputs, dtype=np.int32)  # ✅ int32, not float32 (embedding layer needs integers)

def predict_fn(input_data, model):
    preds = model.predict(input_data)
    return preds

def output_fn(prediction, content_type):
    return json.dumps({"predictions": prediction.tolist()})  # ✅ wrap in dict