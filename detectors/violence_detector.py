import tensorflow as tf
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
from config import TFLITE_MODEL_PATH

# Load and initialize the TFLite model
interpreter = Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
runner = interpreter.get_signature_runner()

# Initialize states for the TFLite model
init_states = {
    name: tf.zeros(x['shape'], dtype=x['dtype']).numpy()
    for name, x in runner.get_input_details().items()
}
del init_states['image']

def detect_violence(frame, states):
    processed_frame = tf.image.convert_image_dtype(frame, tf.float32)
    processed_frame = tf.image.resize_with_pad(processed_frame, 172, 172)
    clip = tf.expand_dims(processed_frame, axis=0)
    
    outputs = runner(**states, image=clip)
    logits = outputs.pop('logits')[0]
    states = {key: value for key, value in outputs.items()}
    
    # Calculate softmax probabilities
    probs = tf.nn.softmax(logits).numpy()
    anomaly = 1 if probs[0] > 0.6 else 0  # Threshold at 80%
    return anomaly, states,probs
