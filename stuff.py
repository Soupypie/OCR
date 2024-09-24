import tensorflow as tf
impoor

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)
    
import tensorflow as tf
print(tf.test.is_built_with_cuda())
