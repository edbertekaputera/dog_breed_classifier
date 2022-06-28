# Imported Libraries
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os


#Function for preprocessing
def process_image(filename):
  """
  Takes an image filepath and turns it into a tensor.
  """
  #read image
  image = tf.io.read_file(filename)

  #turn jpeg to numerical Tensor with 3 color channels (RGB)
  image = tf.image.decode_jpeg(image, channels=3)

  #Convert colour channels values 0-255 to 0-1 values.
  #This is a normalization process to make process more efficient.
  image = tf.image.convert_image_dtype(image, tf.float32)

  #Resize to (224,224)
  image = tf.image.resize(image, size=[224, 224])

  return image

#Create a function to return a tuple of (image, label)
def get_image_tuple(filename, label):
  """
  Takes an image file path and label
  then processes and return a tuple (image, label)
  """
  image = process_image(filename)
  return image, label


#function to turn data to batches
def create_data_batches(x, y=None, batch_size=32, valid_data=False, test_data=False):
  """
  Create batches of data out of (image x) and (label y) pairs.
  Shuffles the data if its training data, but not when validation data.
  Also accepts test data as input (no labels).
  """
  #If test dataset, we probably don't have labels
  if test_data:
    print(f"Creating test data batches... BATCH SIZE={batch_size}")
    data = tf.data.Dataset.from_tensor_slices(tf.constant(x))
    data_batch = data.map(process_image).batch(batch_size)

  #If validation dataset, we don't need to shuffle
  elif valid_data:
    print(f"Creating validation data batches... BATCH SIZE={batch_size}")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), #filepath
                                              tf.constant(y))) # labels
    data_batch = data.map(get_image_tuple).batch(batch_size)
  
  else:
    print(f"Creating training data batches... BATCH SIZE={batch_size}")
    #turn filepath and labels into Tensors
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), #filepath
                                              tf.constant(y))) # labels
    #shuffle
    data = data.shuffle(buffer_size=len(x))

    #process into (image,label) tuple and make batch
    data = data.map(get_image_tuple)
    data_batch = data.batch(batch_size)
  return data_batch

# Load model function
def load_model(model_path):
  """
  Loads a saved model from a specified path
  """
  print(f"Loading saved model from: {model_path}")
  model = tf.keras.models.load_model(model_path,
                                     custom_objects={"KerasLayer" : hub.KerasLayer})
  return model

# Check accuracy of prediction
def preds_check(preds, verbose=False):
   labels = []
   for folders in os.listdir("Images/train_images"):
      labels.append("".join(folders.split("-")[1:]))
   unique_breeds = np.unique(labels)
   print(unique_breeds)
   if verbose:
      print(preds)
   print(f"Max value (probability of prediction) : {np.max(preds[0])}")
   print(f"Sum : {np.sum(preds[0])}")
   print(f"Max Index: {np.argmax(preds[0])}")
   print(f"Predicted label: {unique_breeds[np.argmax(preds[0])]}")
  
# COMPLETE DOG BREED PREDICTION FUNCTION
def dog_breed_predict(filename_list: list, model):
   # Turn images into batch datasets
   file_batch = create_data_batches(filename_list, test_data=True)
   # Make predictions on the data
   prediction = model.predict(file_batch)
   # Check predictions
   preds_check(prediction)