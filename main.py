# Imported Libraries
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
from func import load_model, dog_breed_predict

def main():
   model = load_model("Models/2022_06_15-16_001655308824-full-image-set-mobilenetv2-Adam.h5")
   while(True):
      image = [input("Enter the image path: ")]
      dog_breed_predict(image, model)
      if (input("Try again? (y/n) ").lower() == "n"):
         print("QUITTING...")
         break

if __name__ == "__main__":   
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
   print("DOG BREED PREDICTION PROGRAM...")
   main()