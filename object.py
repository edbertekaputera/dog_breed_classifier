# Imported Libraries
from cv2 import FONT_HERSHEY_DUPLEX, FONT_HERSHEY_PLAIN
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import cv2
import argparse
from time import time
import matplotlib.pyplot as plt
# from main import DogModel

# ObjectDetectionModel Class
class ObjectDetectModel:
   # Class attributes
   __OBJECT_MODEL_PATH = "ssd_mobilenet_v2_2"

   # Special dunder methods
   def __init__(self) -> None:
      self.__object_model = self.__loadModel()
   
   def __str__(self) -> str:
      return "SSD MobileNetV2 Object Detection Model"
   
   # Static image processing methods
   @staticmethod
   def process_whole_image(image):
      """
      Processes the image into a RGB tensor with values ranging from 0-1, 
      Parameters:
      image = Image to be processed;
      """
      processed_image = tf.image.convert_image_dtype(image, tf.uint8)
      return np.array([processed_image])

   # Private Instance methods
   def __loadModel(self):
      return tf.keras.models.load_model(self.__OBJECT_MODEL_PATH)

   # Public Instance methods
   def detect_object(self, image):
      """
      This function processes the image with the `process_whole_image` function,
      then detects the objects within the processed image using SSD-MobileNet V2 model,
      and returns (class_id, boxes, scores).
      -> class_ids: tf.int tensor of shape [N] containing detection class index from the label file.
      -> boxes: tf.float32 tensor of shape [N, 4] containing bounding box coordinates in the following order: [ymin, xmin, ymax, xmax].
      -> scores: a tf.float32 tensor of shape [N] containing detection scores.
      
      Parameters:
      image: Image to be detected.
      """
      img = self.process_whole_image(image)
      results = self.__object_model(img)
      results = {key:value.numpy() for key,value in results.items()}
      class_ids = results["detection_classes"][0]
      boxes = results["detection_boxes"][0]
      scores = results["detection_scores"][0]
      return class_ids, boxes, scores

   def getDog(self, image):
      """
      This function uses the `process_whole_image` and `detect` to process and detect dogs in the image,
      then returns (detection_score, coordinates). If no dogs are found, returns (-1, -1).
      detection_score = confidence score of the detection.
      coordinates = two tuples (xmin, ymin), (xmax, ymax) representing both corner coordinates of the bounding box.
      
      Parameters:
      image = Image to be detected.
      """
      im_height, im_width = image.shape[:2]
      class_ids, boxes, scores = self.detect_object(image)

      for i in range(min(boxes.shape[0], 10)):
         if scores[i] >= 0.1 and class_ids[i] == 18:
            xmin = int(boxes[i][1] * im_width) - 10 if int(boxes[i][1] * im_width) - 10 > 0 else 0
            xmax = int(boxes[i][3] * im_width) + 10 if int(boxes[i][3] * im_width) + 10 < im_width else im_width
            ymin = int(boxes[i][0] * im_height) - 10 if int(boxes[i][0] * im_height) - 10 > 0 else 0
            ymax = int(boxes[i][2] * im_height) + 10 if int(boxes[i][2] * im_height) + 10 < im_height else im_height
            coordinates = ((xmin, ymin), (xmax, ymax))
            return scores[i], coordinates
      return -1, -1

   def getDogCrop(self, image):
      scores, coordinates = self.getDog(image)
      newimg = -1
      if scores != -1:
         xmin = coordinates[0][0]
         xmax = coordinates[1][0]
         ymin = coordinates[0][1]
         ymax = coordinates[1][1]
         newimg = image[ymin:ymax, xmin:xmax]
         return newimg, ((xmin, ymin), (xmax, ymax))
      else:
         return -1, -1

def main():
   detect_model = ObjectDetectModel()
   ap = argparse.ArgumentParser()
   ap.add_argument("-f", "--frame_tick", type=int, help="detects every x frames.", default=1)
   video = cv2.VideoCapture(1)
   args = vars(ap.parse_args())
   frame_tick = args["frame_tick"]
   counter = 0
   RED = (0,0,255)
   WHITE = (255, 255, 255)
   box_list = []
   text_list = []
   while True:
      ret, frame = video.read()
      if counter == frame_tick:
         box_list = []
         text_list = []
         scores, coordinates = detect_model.getDog(frame)
         if scores != -1:
            box_list.append(coordinates)
            text_list.append(f"DOG {scores*100:.2f}%")
         counter = 0
      counter+= 1
      #Waits for a user input to quit the application
      if len(box_list) > 0:
         pos1 = box_list[0][0]
         pos2 = box_list[0][1]
         pos_text = (pos1[0]+12, pos1[1]+12)
         pos_text_rect_1 = (pos1[0], pos1[1])
         pos_text_rect_2 = (pos2[0], pos1[1]+15)
         cv2.rectangle(frame, pos1, pos2, RED, 4)
         cv2.rectangle(frame, pos_text_rect_1, pos_text_rect_2, RED, -1)
         cv2.putText(frame,  text_list[0], pos_text, FONT_HERSHEY_DUPLEX, 0.5, WHITE)
      cv2.imshow("test", frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
         break


if __name__ == "__main__":
   main()