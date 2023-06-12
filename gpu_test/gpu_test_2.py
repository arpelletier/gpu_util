import tensorflow as tf
import nvidia_smi
import time

# Define a simple TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1000, input_shape=(1000,))
])

# Generate some random input data
data = tf.random.normal((1000, 1000))

# Initialize the NVML library
nvidia_smi.nvmlInit()

# Run the model without GPU acceleration
start_time = time.time()
with tf.device('/CPU:0'):
    output = model(data)
end_time = time.time()
cpu_time = end_time - start_time

# Run the model with GPU acceleration (if available)
if tf.config.experimental.list_physical_devices('GPU'):
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    gpu_device = gpu_devices[0]  # Assuming only one GPU is available
    gpu_device_name = gpu_device.name
    gpu_device_index = int(gpu_device_name.split(':')[-1])
    gpu_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_device_index)

    with tf.device(gpu_device_name):
        start_time = time.time()
        output = model(data)
        end_time = time.time()
        gpu_time = end_time - start_time

        # GPU utilization
        utilization = nvidia_smi.nvmlDeviceGetUtilizationRates(gpu_handle)
        gpu_utilization = utilization.gpu
        print("GPU utilization: {:.2f}%".format(gpu_utilization))

else:
    gpu_time = None

# Print the results
print("CPU time: {:.2f} seconds".format(cpu_time))
if gpu_time:
    print("GPU time: {:.2f} seconds".format(gpu_time))
    print("Speedup: {:.2f}x".format(cpu_time / gpu_time))
else:
    print("No GPU available.")

# Shutdown the NVML library
nvidia_smi.nvmlShutdown()

