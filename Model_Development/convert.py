import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("Trained_Models/2022_08_18-12_49_1660826996-MobileNetV2_100_Adam_20580_Augmented") # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)