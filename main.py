# Imported libraries
from dog import DogModel
from flask import Response, Flask, render_template
import cv2
import threading
import argparse



# Initialize outputFrame and thread lock
outputFrame = None
lock = threading.Lock()

# Initialize Flask
app = Flask(__name__)

# Initialize Camera
video = cv2.VideoCapture(1)

@app.route("/")
def index():
   return render_template("index.html")

@app.route("/video_feed")
def video_feed():
   return Response(generate(), 
                  mimetype="multipart/x-mixed-replace; boundary=frame")

# Video Detection function
def dog_detect(frame_tick=60):
   global outputFrame, video, lock
   # Initialize Model
   model_path = "breed_model/Models/2022_06_29-07_471656488859-full-image-set-mobilenetv2-Adam.h5"
   my_model = DogModel(model_path)
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
      with lock:
         outputFrame = frame.copy()

def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock
	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue
			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
			# ensure the frame was successfully encoded
			if not flag:
				continue
		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

if __name__ == "__main__":
   # Construct argument parser for arg in commandline
   ap = argparse.ArgumentParser()
   # Argument for IP
   ap.add_argument("-i", "--ip", type=str, default="127.0.0.1",
		help="ip address of the device")
   # Argument for port
   ap.add_argument("-o", "--port", type=int, default=8000,
		help="ephemeral port number of the server (1024 to 65535)")
   # Argument for frame ticker
   ap.add_argument("-f", "--frame-count", type=int, default=60,
		help="# of frames used to construct the background model")
   
   args = vars(ap.parse_args())

   # start a thread that will perform motion detection
   t = threading.Thread(target=dog_detect, args=(
		args["frame_count"],))
   t.daemon = True
   t.start()
	
   # start the flask app
   app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)
