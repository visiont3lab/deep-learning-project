import cv2
from PIL import Image
import pickle

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

class MaskDetector:
  def __init__(self,modelPath,modelMaskPath):
    self.cl = cv2.CascadeClassifier(modelPath)
    with open(modelMaskPath, 'rb') as f:
      self.mask_model = pickle.load(f)

    self.mask_model_names = ["mask", "no-mask"]
  def run(self,imageRGB):
    # we assume RGB image as input
    gray = cv2.cvtColor(imageRGB, cv2.COLOR_RGB2GRAY)
    faces = self.cl.detectMultiScale( gray, scaleFactor=1.1, minNeighbors=5,minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    imDisplay = imageRGB.copy()
    for (x, y, w, h) in faces:
      # Process
      imProcess = imageRGB.copy()
      roi = imProcess[y:y+h,x:x+w]

      ## Prediction
      x = cv2.resize(roi,64,64)
      x = x.reshape(-1)
      res = self.mask_model.predict(roi) # 0 mask , 1 no mask

      cv2.putText(imDisplay,self.mask_model_names[res] , (10,10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3, cv2.LINE_AA)

      cv2.rectangle(imDisplay, (x,y), (x+w,y+h) ,(0, 255, 0), 2)
    return imDisplay