import cv2
from PIL import Image

class FaceDetector:
  def __init__(self,modelPath):
    self.cl = cv2.CascadeClassifier(modelPath)
  def run(self,imageRGB):
    # we assume RGB image as input
    gray = cv2.cvtColor(imageRGB, cv2.COLOR_RGB2GRAY)
    faces = self.cl.detectMultiScale( gray, scaleFactor=1.05, minNeighbors=5,minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    imDisplay = imageRGB.copy()
    for (x, y, w, h) in faces:
      # Process
      imProcess = imageRGB.copy()
      roi = imProcess[y:y+h,x:x+w]
      cv2.rectangle(imDisplay, (x,y), (x+w,y+h) ,(0, 255, 0), 2)
    return imDisplay