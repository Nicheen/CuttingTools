import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

execution_times = {}

def timer(func): 
    def wrapper(*args, **kwargs):
        start_time = cv2.getTickCount() # Get the start time
        result = func(*args, **kwargs) # Run the function which is getting timed
        end_time = cv2.getTickCount() # Get the end time
        raw_elapsed_time = (end_time - start_time)/cv2.getTickFrequency()
        elapsed_time = raw_elapsed_time
        
        if raw_elapsed_time < 1e-6: # Less than 1 microsecond
            elapsed_time *= 1e9 # Convert to nanosecond
            time_unit = "ns"
        elif raw_elapsed_time < 1e-3:  # Less than 1 millisecond
            elapsed_time *= 1e6  # Convert to microseconds
            time_unit = "µs"
        elif raw_elapsed_time < 1:  # Less than 1 second
            elapsed_time *= 1e3  # Convert to milliseconds
            time_unit = "ms"
        else:
            time_unit = "s"  # Keep in seconds
        
        if func.__name__ not in execution_times:
            execution_times[func.__name__] = []
        execution_times[func.__name__].append(raw_elapsed_time)
            
        print(f"Execution time for {func.__name__}: {elapsed_time:.2f} {time_unit}") # Print out time in console
        return result
    return wrapper

def combine_post_processing(exec_times):
    """Combine execution times for specific functions into a single 'post_processing' category."""
    post_processing_keys = ['seperateGrid', 'thresholdCuttingTools', 'countWhitePixels']
    post_processing_times = []

    # Combine the times and remove the individual entries
    for key in post_processing_keys:
        if key in exec_times:
            post_processing_times.extend(exec_times.pop(key))

    # Add the combined category
    if post_processing_times:
        exec_times['post_processing'] = post_processing_times

def plot_execution_times(skip_warmup=10):
    combine_post_processing(execution_times)
    
    names = []
    averages = []
    errors = []
    
    # Collect data for plotting
    for name, durations in execution_times.items():
        filtered_durations = durations[skip_warmup:]
        if filtered_durations:
            names.append(name)
            average =np.mean(filtered_durations)
            averages.append(average)
            min_val = np.min(filtered_durations)
            max_val = np.max(filtered_durations)
            errors.append([average - min_val, max_val - average])
        
    errors = np.array(errors).T

    # Create a plot
    fig, ax = plt.subplots()
    bars = ax.bar(names, averages, color='skyblue', yerr=errors, capsize=5, label='Average')
    ax.set_xlabel('Function Name')
    ax.set_ylabel('Execution Time (s)')
    ax.set_title('Average Execution Time of Functions with Min/Max Ranges')
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.show()  


def show(image, path=None):
    if path:
        cv2.imwrite(path, image)
    else:
        cv2.imshow("Test", image)
        cv2.moveWindow("Test", 100, 100)
        cv2.waitKey(0)
        cv2.destroyWindow("Test")
          
def seperateGrid(image, thickness):
    
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Define the ROI (Region of Interest) excluding 20 pixels from each side
    x_start = 20
    y_start = 20
    x_end = width - 20
    y_end = height - 20

    # Crop the image using array slicing
    image = image[y_start:y_end, x_start:x_end]
    
    # Get the image size in pixels
    initial_height, initial_width = image.shape[:2]
    
    # Create an empty image with the same size as the image to later become the grid
    gridmask = np.ones((initial_height, initial_width), dtype='uint8')

    # Define how many tiles are in the container
    num_vertical_tiles = 2
    num_horizontal_tiles = 5
    
    tile_height = initial_height // num_vertical_tiles - (2 * thickness) 
    tile_width = initial_width // num_horizontal_tiles - (2 * thickness)
    
    tiles = []
    for y in range(num_vertical_tiles):
        for x in range(num_horizontal_tiles):
            y_start = y * (initial_height // num_vertical_tiles) + thickness
            y_end = y_start + tile_height - thickness
            x_start = x * (initial_width // num_horizontal_tiles) + thickness
            x_end = x_start + tile_width - thickness
            
            tiles.append(image[y_start:y_end, x_start:x_end])
            
            cv2.rectangle(gridmask, (x_start, y_start), (x_end, y_end), color=0, thickness=-1)
    
    grid = cv2.bitwise_and(image, image, mask=gridmask)
    
    return grid, gridmask, tiles

def thresholdCuttingTools(tiles, image, adaptive=False):
    # Define how many tiles are in the container
    num_vertical_tiles = 2
    num_horizontal_tiles = 5
    
    i = 0
    filled_contours = []
    for y in range(num_vertical_tiles):
        row = []
        for x in range(num_horizontal_tiles):
            tile = tiles[i]
            blank = np.zeros(tile.shape[:2], dtype='uint8')
            
            if adaptive:
                blocksize = 25
                c = -5
                threshold = cv2.adaptiveThreshold(tile, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, (blocksize*2)+1, c)
            else:
                _, threshold = cv2.threshold(tile, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))
            dilated = cv2.dilate(threshold, kernel)
            
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
            
            hull_largest = cv2.convexHull(largest_contour, False)
        
            filled_contour = cv2.drawContours(blank, [hull_largest], -1, color=255, thickness=cv2.FILLED)
        
            row.append(filled_contour)
            i += 1
        filled_contours.append(row)

    filled_contours_image = concat_tile_2d(filled_contours)

    combine = cv2.bitwise_or(image, image, mask=filled_contours_image)
    rgbImage = cv2.cvtColor(combine, cv2.COLOR_GRAY2RGBA)
    
    return rgbImage, filled_contours

def countWhitePixels(tiles, finalImage):
    # Define how many tiles are in the container
    num_vertical_tiles = 2 
    num_horizontal_tiles = 5
    
    tiles = flatten_2d_array(tiles)
    white_counts = [np.sum(tile == 255) for tile in tiles]
    
    white_median = np.median(white_counts)
    i = 0
    
    passed = True
    for y in range(num_vertical_tiles):
        for x in range(num_horizontal_tiles):
            tile_height, tile_width = tiles[i].shape[:2]
            text_x = x * tile_width
            text_y = (y + 1) * tile_height - 10
            nWhite = white_counts[i]
            
            procent = int(nWhite / white_median * 100)
            
            if 60 <= procent <= 120:
                finalImage = cv2.putText(finalImage, f'{nWhite} ({procent}%)', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                passed = False
                finalImage = cv2.putText(finalImage, f'{nWhite} ({procent}%)', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            i += 1
           
    return passed, finalImage

def heightInspection(image):
    toohigh = np.sum(image >= 255) > 0 # Check if cutting tool is too high for lid attachment
    if toohigh:
        points = np.argwhere(image >= 255)
        for point in points:
            return toohigh, point
    else:
        return tooHigh, -1, -1

def concat_tile_2d(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

def concat_tile(im_list):
    # Define how many tiles are in the container
    num_vertical_tiles = 2
    num_horizontal_tiles = 5
    
    # Split the image list into a 2D list of tiles
    im_list_2d = [im_list[i:i+num_horizontal_tiles] for i in range(0, len(im_list), num_horizontal_tiles)]
    
    # Concatenate the tiles
    return concat_tile_2d(im_list_2d)

def flatten_2d_array(arr_2d):
    return [element for sublist in arr_2d for element in sublist]

