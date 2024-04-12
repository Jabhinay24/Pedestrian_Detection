from tensorflow.keras.models import load_model
import cv2
import imutils

def load_model_and_hog():
  
  """Loads the pre-trained model and initializes the HOG detector."""
  # Load the pre-trained model

  model = load_model('Pedestrian_Detection.h5')
  print("Pedestrian detection model loaded successfully!")

  # Initialize the HOG detector

  hog = cv2.HOGDescriptor()
  hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

  return model, hog

def detect_pedestrians(image, model, hog):
  
  """Detects potential pedestrian regions in the image."""
  # Resize the image to a maximum width of 400 pixels

  image = imutils.resize(image, width=400)

  # Detect pedestrians

  (regions, _) = hog.detectMultiScale(image,winStride=(4, 4),padding=(4, 4),scale=1.05)

  return regions

def draw_bounding_boxes(image, regions):
  
  """Draws bounding boxes around detected pedestrian regions."""
  # Draw bounding boxes

  for (x, y, w, h) in regions:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

  return image

def display_and_close(image):
  """Displays the image and closes windows."""

  cv2.imshow("Image", image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

# Main program
if __name__ == "__main__":
  # Load model and HOG detector
  model, hog = load_model_and_hog()

  # Read the image
  image = cv2.imread("C:\\Users\\Abhinay\\OneDrive\\Documents\\Pedestrian_Detection\\data\\validation\\pedestrian\\val (161).jpg")

  # Detect pedestrians
  regions = detect_pedestrians(image.copy(), model, hog)  # Avoid modifying the original image

  # Draw bounding boxes
  image_with_boxes = draw_bounding_boxes(image.copy(), regions)

  # Display and close
  display_and_close(image_with_boxes)




