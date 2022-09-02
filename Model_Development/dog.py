# Imported Libraries
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import cv2
import argparse
from object import ObjectDetectModel

class DogModel(ObjectDetectModel):
   # Private class attribute
   __defaultPath = "./Models/TFLite_Models/mobilenetV3_adam_trained.tflite"
   __classes = ['Afghan_hound', 'African_hunting_dog', 'Airedale', 'American_Staffordshire_terrier', 'Appenzeller', 'Australian_terrier', 'Bedlington_terrier', 'Bernese_mountain_dog', 'Blenheim_spaniel', 'Border_collie', 'Border_terrier', 'Boston_bull', 'Bouvier_des_Flandres', 'Brabancon_griffon', 'Brittany_spaniel', 'Cardigan', 'Chesapeake_Bay_retriever', 'Chihuahua', 'Dandie_Dinmont', 'Doberman', 'English_foxhound', 'English_setter', 'English_springer', 'EntleBucher', 'Eskimo_dog', 'French_bulldog', 'German_shepherd', 'German_shorthaired_pointer', 'Gordon_setter', 'Great_Dane', 'Great_Pyrenees', 'Greater_Swiss_Mountain_dog', 'Ibizan_hound', 'Irish_setter', 'Irish_terrier', 'Irish_water_spaniel', 'Irish_wolfhound', 'Italian_greyhound', 'Japanese_spaniel', 'Kerry_blue_terrier', 'Labrador_retriever', 'Lakeland_terrier', 'Leonberg', 'Lhasa', 'Maltese_dog', 'Mexican_hairless', 'Newfoundland', 'Norfolk_terrier', 'Norwegian_elkhound', 'Norwich_terrier', 'Old_English_sheepdog', 'Pekinese', 'Pembroke', 'Pomeranian', 'Rhodesian_ridgeback', 'Rottweiler', 'Saint_Bernard', 'Saluki', 'Samoyed', 'Scotch_terrier', 'Scottish_deerhound', 'Sealyham_terrier', 'Shetland_sheepdog', 'ShihTzu', 'Siberian_husky', 'Staffordshire_bullterrier', 'Sussex_spaniel', 'Tibetan_mastiff', 'Tibetan_terrier', 'Walker_hound', 'Weimaraner', 'Welsh_springer_spaniel', 'West_Highland_white_terrier', 'Yorkshire_terrier', 'affenpinscher', 'basenji', 'basset', 'beagle', 'blackandtan_coonhound', 'bloodhound', 'bluetick', 'borzoi', 'boxer', 'briard', 'bull_mastiff', 'cairn', 'chow', 'clumber', 'cocker_spaniel', 'collie', 'curlycoated_retriever', 'dhole', 'dingo', 'flatcoated_retriever', 'giant_schnauzer', 'golden_retriever', 'groenendael', 'keeshond', 'kelpie', 'komondor', 'kuvasz', 'malamute', 'malinois', 'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer', 'otterhound', 'papillon', 'pug', 'redbone', 'schipperke', 'silky_terrier', 'softcoated_wheaten_terrier', 'standard_poodle', 'standard_schnauzer', 'toy_poodle', 'toy_terrier', 'vizsla', 'whippet', 'wirehaired_fox_terrier']

   # Constructors
   def __init__(self, model_path) -> None:
      super().__init__()
      # Private instances
      self.__breed_model = self.__load_model(model_path)
   # Alternative Constructor (default)
   @classmethod
   def init_default(cls):
      """
      Constructs an instance of DogModel with default model path
      """
      return cls(cls.__defaultPath)

   # Private static method
   @staticmethod
   def __load_model(model_path):
      """
      Loads a saved model from a specified path
      """
      print(f"Loading TFLite model from: {model_path}")
      model = tf.lite.Interpreter(model_path)
      model.allocate_tensors()
      return model

   @classmethod
   def classes(cls):
      return cls.__classes

   # Public getter/setter method for the model
   @property
   def breed_model(self):
      return self.__breed_model
   @breed_model.setter
   def set_breed_model(self, path):
      """
      Sets the breed model into a new model which would be loaded from the path.
      Parameters:
      path = path of the saved model
      """
      self.__breed_model = self.__load_model(path)

   # Public instance method
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
      return np.array([processed_image]), coordinates
      
   # Check accuracy of prediction
   def predict(self, image, check=False, verbose=False):
      processed_image, coordinate = self.process_image(image)
      if coordinate == -1:
         return -1, -1
      
      input_details = self.__breed_model.get_input_details()
      output_details = self.__breed_model.get_output_details()
      self.__breed_model.set_tensor(input_details[0]['index'], processed_image)
      self.__breed_model.invoke()
      self.prediction_probability = self.__breed_model.get_tensor(output_details[0]['index'])
      if check:
         breed = self.predict_check(verbose)
         return coordinate, breed
      else:
         return coordinate
      
   def predict_check(self, verbose=False):
      if verbose:
         print(self.prediction_probability)
      print(np.argmax(self.prediction_probability[0]))
      score = np.max(self.prediction_probability[0])
      print(f"Max value (probability of prediction) : {score}")
      print(f"Sum : {np.sum(self.prediction_probability[0])}")
      print(f"Max Index: {np.argmax(self.prediction_probability[0])}")
      label = DogModel.__classes[np.argmax(self.prediction_probability[0])]
      print(f"Predicted label: {label}")
      return (label, score)

def main():
   ap = argparse.ArgumentParser()
   ap.add_argument("-p", "--path", type=str,help="path of the saved model.")
   ap.add_argument("-f", "--frame_tick", type=int, help="detects every x frames.", default=2)
   args = vars(ap.parse_args())
   model_path = args["path"]
   frame_tick = args["frame_tick"]
   if model_path == None:
      print("Missing Path Argument...")
      print("Using default Init")
      my_model = DogModel.init_default()
   else:
      my_model = DogModel(model_path)
   video = cv2.VideoCapture(1)
   counter = 0
   RED = (0,0,255)
   WHITE = (255, 255, 255)
   box_list = []
   text_list = []
   while True:
      ret, frame = video.read()
      if counter == frame_tick:
         coordinates, breed = my_model.predict(frame, check=True)
         box_list = []
         text_list = []
         if coordinates != -1:
            box_list.append(coordinates)
            text_list.append(f"{breed[0]} {breed[1]*100:.2f}%")
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

