import tensorflow as tf
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print("Detected GPUs:")
else:
    print("No Detectable GPU's")
        
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
