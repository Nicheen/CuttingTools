import os                                                                       # Library to handle Windows pathing and file creation
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'                                       # Fixes annoying warning message on some computers

import cv2 
import torch                                                                    # Library that handles the tensors (is needed for DepthAnything)
import numpy as np                                                              # numpy is Pythons math library which contains alot of matrix operations
from transformers import AutoImageProcessor, AutoModelForDepthEstimation        # Transformers library that handle the AI model DepthAnything
from opencvFunctions import *

def _load_image(path):
    size=(1, 1); reduce=55
    image = cv2.imread(path)
    name = path.split("/")[-1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 35, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(big_contour)
    #n x, y, w, h = 450, 950, 3681-450, 2285-950
    crop = image[y:y+h, x:x+w]
    
    # Get the image size in pixels
    initial_height, initial_width = crop.shape[:2]
    
    # Reducing the image borders
    border_reduction = 15
    crop = crop[border_reduction:initial_height-border_reduction, border_reduction:initial_width-border_reduction]

    return image, name, crop

class DepthAnything:
    def __init__(self, modelSize='large'):
        # Here we pick which device the AI model will run on, either the cpu or gpu ("cuda").
        # NOTE: if you have a gpu and it still run as "cpu", it's most likely a compatability issue with libraries
        # Check that your torch version is compatible with your cuda version!
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_processor = AutoImageProcessor.from_pretrained(f"LiheYoung/depth-anything-{modelSize}-hf")
        self.model = AutoModelForDepthEstimation.from_pretrained(f"LiheYoung/depth-anything-{modelSize}-hf").to(self.device)
        print(f'Loaded the "{modelSize}" model, using {self.device}. (cuda / cpu)')
        print(f'OpenCV Optimized? : {cv2.useOptimized()}')
    
    def load_image(self, path):
        self.image, self.name, self.crop = _load_image(path)

    def predict(self):
        inputs = self.image_processor(images=self.crop, return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.predicted_depths = self.model(**inputs).predicted_depth
            
    def formatImage(self, constant_size=(2072,784)):
        formats = self.predicted_depths.numpy(force=True)
        self.output = (formats * 255 / np.max(formats)).astype('uint8')[0]
        self.output_resized = cv2.resize(self.output, constant_size)
        return self.output_resized


class MiDaS:
    def __init__(self, modelSize="DPT_BEiT_L_512"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load('intel-isl/MiDaS', modelSize)
        self.model.to(self.device)
        print(f'Loaded the "{modelSize}" model, using {self.device}. (cuda / cpu)')
        print(f'OpenCV Optimized? : {cv2.useOptimized()}')
        
        self.transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
        if modelSize == "DPT_BEiT_L_512":
            self.transform = self.transforms.beit512_transform
        elif modelSize in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = self.transforms.dpt_transform
        else:
            self.transform = self.transforms.small_transform
    
    def load_image(self, path):
        self.image, self.name, self.crop = _load_image(path)

    def predict(self):
        self.crop = cv2.cvtColor(self.crop, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(self.crop).to(self.device)
        with torch.no_grad():
            self.predicted_depths = self.model(input_batch)
            
    def formatImage(self, inverted=False):
        formats = self.predicted_depths.numpy(force=True)
        self.output = (formats * 255 / np.max(formats)).astype('uint8')[0]
        return self.output

