# Imported Libraries
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import cv2
from object import ObjectDetectModel

class DogModel(ObjectDetectModel):
   def __init__(self, model_path) -> None:
      super().__init__()
      # Declaring instances
      self.breed_model = None
   
      # Loads model
      self.load_model(model_path)

   def load_model(self, model_path):
      """
      Loads a saved model from a specified path
      """
      print(f"Loading saved model from: {model_path}")
      model = tf.keras.models.load_model(model_path,
                                          custom_objects={"KerasLayer" : hub.KerasLayer})
      self.breed_model = model
   
   def process_image(self, image, length=224, width=224):
      """
      Processes the image into a 224x224x3 tensor with values ranging from 0-1, 
      then converts it to a batch for prediction
      """
      new_image, coordinates = self.getDogCrop(image)
      if coordinates == -1:
         return -1, -1
      processed_image = tf.image.convert_image_dtype(new_image, tf.float32)
      #Resize to (224,224)
      processed_image = tf.image.resize(processed_image, size=[length, width])
      return tf.data.Dataset.from_tensor_slices([processed_image]).batch(32), coordinates
      
   # Check accuracy of prediction
   def predict(self, image, check=False):
      processed_image, coordinate = self.process_image(image)
      if processed_image == -1:
         return -1, -1
      self.prediction_probability = self.breed_model.predict(processed_image)
      if check:
         breed = self.predict_check()
         return coordinate, breed
      else:
         return coordinate
      
   def predict_check(self, verbose=False):
      labels = []
      for folders in os.listdir("Images/train_images"):
         labels.append("".join(folders.split("-")[1:]))
      unique_breeds = np.unique(labels)
      if verbose:
         print(unique_breeds)
         print(self.prediction_probability)
      label = unique_breeds[np.argmax(self.prediction_probability[0])]
      score = np.max(self.prediction_probability[0])
      print(f"Max value (probability of prediction) : {score}")
      print(f"Sum : {np.sum(self.prediction_probability[0])}")
      print(f"Max Index: {np.argmax(self.prediction_probability[0])}")
      print(f"Predicted label: {label}")
      return (label, score)

def main():
   model_path = "full_model.hdf5"
   my_model = DogModel(model_path)
   video = cv2.VideoCapture(1)
   counter = 0
   RED = (0,0,255)
   WHITE = (255, 255, 255)
   box_list = []
   text_list = []
   while True:
      ret, frame = video.read()
      if counter == 60:
         box_list = []
         text_list = []
         coordinates, breed = my_model.predict(frame, check=True)
         if coordinates != -1:
            box_list.append(coordinates)
            text_list.append(f"DOGBREED = {breed[0]} {breed[1]*100:.2f}%")
         counter = 0
      counter+= 1   

      # Draws all bounding boxes
      for i in range(len(box_list)):
         pos1 = box_list[i][0]
         pos2 = box_list[i][1]
         pos_text = (pos1[0]+12, pos1[1]+12)
         pos_text_rect_1 = (pos1[0], pos1[1])
         pos_text_rect_2 = (pos2[0], pos1[1]+15)
         cv2.rectangle(frame, pos1, pos2, RED, 4)
         cv2.rectangle(frame, pos_text_rect_1, pos_text_rect_2, RED, -1)
         cv2.putText(frame,  text_list[0], pos_text, cv2.FONT_HERSHEY_DUPLEX, 0.5, WHITE)
      cv2.imshow("test", frame)
      
      #Waits for a user input to quit the application
      if cv2.waitKey(1) & 0xFF == ord('q'):
         break

if __name__ == "__main__":   
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
   print("DOG BREED PREDICTION PROGRAM...")
   main()