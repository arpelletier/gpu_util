import tensorflow as tf
import time

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# Define a simple TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1000, input_shape=(1000,))
])

# Generate some random input data
data = tf.random.normal((1000, 1000))

# Run the model without GPU acceleration
start_time = time.time()
with tf.device('/CPU:0'):
    output = model(data)
end_time = time.time()
cpu_time = end_time - start_time

# Run the model with GPU acceleration (if available)
if tf.config.experimental.list_physical_devices('GPU'):
    start_time = time.time()
    with tf.device('/GPU:0'):
        output = model(data)
    end_time = time.time()
    gpu_time = end_time - start_time
else:
    gpu_time = None

# Print the results
print("CPU time: {:.2f} seconds".format(cpu_time))
if gpu_time:
    print("GPU time: {:.2f} seconds".format(gpu_time))
    print("Speedup: {:.2f}x".format(cpu_time / gpu_time))
else:
    print("No GPU available.")

