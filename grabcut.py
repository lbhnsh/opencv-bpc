import cv2
import numpy as np

# Global variables to store mouse selection
rect = (0, 0, 0, 0)
drawing = False

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global rect, drawing
    
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button down
        drawing = True
        rect = (x, y, 0, 0)
    
    elif event == cv2.EVENT_MOUSEMOVE:  # Mouse move
        if drawing:
            rect = (rect[0], rect[1], x - rect[0], y - rect[1])
            temp_image = image.copy()
            cv2.rectangle(temp_image, (rect[0], rect[1]), (x, y), (0, 255, 0), 2)
            cv2.imshow('Image', temp_image)
    
    elif event == cv2.EVENT_LBUTTONUP:  # Left mouse button up
        drawing = False
        rect = (rect[0], rect[1], x - rect[0], y - rect[1])
        cv2.rectangle(image, (rect[0], rect[1]), (x, y), (0, 255, 0), 2)
        cv2.imshow('Image', image)

# Load the image
image = cv2.imread('c:/Users/ACER/Downloads/000000rgb.png')  # Replace with your image path

# Resize the image to fit the screen
max_width = 800  # Set maximum width for display
height, width = image.shape[:2]
aspect_ratio = width / float(height)
new_width = max_width
new_height = int(new_width / aspect_ratio)
resized_image = cv2.resize(image, (new_width, new_height))

# Create a mask, background, and foreground models
mask = np.zeros(resized_image.shape[:2], np.uint8)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

# Set up the mouse callback to select the region
cv2.imshow('Image', resized_image)
cv2.setMouseCallback('Image', mouse_callback)

# Wait until a rectangle is selected
cv2.waitKey(0)

# Apply GrabCut after the mouse selection
if rect[2] > 0 and rect[3] > 0:
    # Apply GrabCut algorithm
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Modify the mask to create a binary mask for foreground
    mask2 = np.copy(mask)
    mask2[(mask == 2) | (mask == 0)] = 0  # Background and probable background
    mask2[(mask == 1) | (mask == 0)] = 1  # Foreground and probable foreground
    
    # Extract the foreground using the refined mask
    foreground = image * mask2[:, :, np.newaxis]

    # Resize the foreground to fit the display window
    resized_foreground = cv2.resize(foreground, (new_width, new_height))

    # Display the final foreground image
    cv2.imshow('Foreground', resized_foreground)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally, save the result
    cv2.imwrite('segmented_image.jpg', foreground)
else:
    print("No region selected.")
