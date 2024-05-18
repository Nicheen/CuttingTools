'''
Created by: Gustav Benkowski (gustav.benkowski@gmail.com)
2024-05-18

In the file AiModels can you find the two AI models:
    - Depth Anything
    - MiDaS

In the file opencvFunctions you can find useful functions for cutting tool inspection.
'''

#Libary Imports
import os                                                                       # Library to handle Windows pathing and file creation
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'                                       # Fixes annoying warning message on some computers

import cv2                                                                      # Image library, including edge detection and thresholds with more.
import numpy as np                                                              # numpy is Pythons math library which contains alot of matrix operations
from AiModels import MiDaS, DepthAnything
from opencvFunctions import *
   
def main(DA, path):
    name = os.path.basename(path).split(".")[0]
    DA.load_image(path) # This takes some time so we leave it outside of the timer
    
    start_time = cv2.getTickCount() # Get the start time
    
    DA.predict()
    image = DA.formatImage()
    #show(image, "G:/Output/"+name+".png")
    
    grid, gridmask, tiles = seperateGrid(image, thickness=20)
    #show(grid)
    
    rgbImage, tiles = thresholdCuttingTools(tiles, concat_tile(tiles), adaptive=False)
    
    passedInspection, finalImage = countWhitePixels(tiles, rgbImage)
    
    end_time = cv2.getTickCount() # Get the end time
    elapsed_time = (end_time - start_time)/cv2.getTickFrequency()
    if elapsed_time > 1: finalImage = cv2.putText(finalImage, f'Too slow!', (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 1)
    
    print(f"{DA.name} took {round(elapsed_time, 3)} seconds")
    
    if "F" == name.split("_")[0]: passedInspection = not passedInspection
    
    if not passedInspection:
        #show(finalImage)
        pass
    
    return passedInspection, elapsed_time

def warmup(n, path):
    image = os.path.join(path, os.listdir(path)[0])
    t = [main(DA, image) for _ in range(n)]
    print(f"\n\n\nFinished warmup! (Took {round(sum(t), 3)} seconds)\n\n\n")

if __name__ == "__main__":
    images_path = "G:/Databases/Shiny"
    model = "small"
    DA = DepthAnything(model)
    
    times = []
    passes = []
    n_warmup = 0
    warmup(n_warmup, images_path)
    
    for imageName in os.listdir(images_path):
        if "" in imageName:
            total_path = os.path.join(images_path, imageName)
            passed, time = main(DA, total_path)
            times.append(time)
            passes.append(passed)
            print(passed)
    #plot_execution_times(skip_warmup=n_warmup)
    
    print(f"Number of images in dataset: ({len(times)})")
    print(f"Median Total Time: \t {int(np.median(times)*1000)} ms")
    print(f"Average time: \t\t {int(1000*np.average(times))} ms")
    print(f"Slowest time: \t\t {int(1000*max(times))} ms")
    print(f"Fastest time: \t\t {int(1000*min(times))} ms")
    print(f"Number of images that passed inspection {round(100*sum(passes) / len(passes))}% {model}")
