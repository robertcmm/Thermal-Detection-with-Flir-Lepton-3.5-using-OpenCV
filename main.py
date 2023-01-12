import os
import numpy as np
import cv2
import pandas as pd

base_dir = 'Computer Vision Projects/Thermal Screening Project/'
threshold = 200
area_of_box = 1000        
min_temp = 40.5           
font_scale_caution = 1   
font_scale_temp = 0.7    
save = []

def convert_to_temperature(pixel_avg):
    """
    Converts pixel value (mean) to temperature (fahrenheit) depending upon the camera hardware
    """
    return pixel_avg /2

def color_negative_red(val):
    color = 'red' if val < 0 else 'black'
    return 'color: %s' % color
    
def format_color_groups(values):
    if values >=38.5:
        return  'color:red;border-collapse: collapse; border: 1px solid black;'
    else:
        return  'border-collapse: collapse; border: 1px solid black;'
    

def process_frame(frame):

    
    heatmap_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    heatmap = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_HOT)

    # Binary threshold
    _, binary_thresh = cv2.threshold(heatmap_gray, threshold, 255, cv2.THRESH_BINARY)

    # Image opening: Erosion followed by dilation
    kernel = np.ones((3, 3), np.uint8)
    image_erosion = cv2.erode(binary_thresh, kernel, iterations=1)
    image_opening = cv2.dilate(image_erosion, kernel, iterations=1)

    # Get contours from the image obtained by opening operation
    contours, _ = cv2.findContours(image_opening, 1, 2) 

    image_with_rectangles = np.copy(frame)

    for contour in contours:
        # rectangle over each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Pass if the area of rectangle is not large enough
        if (w) * (h) < area_of_box:
            continue

        # Mask is boolean type of matrix.
        mask = np.zeros_like(heatmap_gray)
        cv2.drawContours(mask, contour, -1, 255, -1)
        # Mean of only those pixels which are in blocks and not the whole rectangle selected
        mean = convert_to_temperature(cv2.mean(heatmap_gray, mask=mask)[0])

        t3mp = (mean-32)*5/9

        
        temperature = round(t3mp,2)
        save.append(temperature)
        print(temperature)
        color = (0, 255, 0) if temperature < min_temp else (
            255, 255, 127)

        # Callback function if the following condition is true
        if temperature >= min_temp:
            # Call back function here
            cv2.putText(image_with_rectangles, "High temperature detected !!!", (35, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_caution, color, 2, cv2.LINE_AA)

        # Draw rectangles for visualisation
        image_with_rectangles = cv2.rectangle(
            image_with_rectangles, (x, y), (x+w, y+h), color, 2)

        # Write temperature for each rectangle
        cv2.putText(image_with_rectangles, "{} C".format(temperature), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_temp, color, 2, cv2.LINE_AA)

    return image_with_rectangles

def main():
    """
    Main driver function
    """
    ## For Video Input
    video = cv2.VideoCapture(1)
    video_frames = []

    while True:
        ret, frame = video.read()
        
        if not ret:
            break
        frame = cv2.resize(frame, (1280,720), interpolation = cv2.INTER_CUBIC)
        # Process each frame
        frame = process_frame(frame)
        height, width, _ = frame.shape
        video_frames.append(frame)
        df = pd.DataFrame(save)
        df = df.style.applymap(format_color_groups)
        df = df.set_table_attributes('style="border-collapse: collapse; border: 1px solid black;"')
        filepath = 'sample2.xlsx'
        
        df.to_excel(filepath, index=False)
        
        # Show the video as it is being processed in a window
        cv2.imshow('frame', frame)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    # Save video to output
    size = (height, width)
    out = cv2.VideoWriter('outputs.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, size)

    for i in range(len(video_frames)):
        out.write(video_frames[i])
    out.release()



if __name__ == "__main__":
    main()
