import cv2
import imutils
from tensorflow.keras.models import load_model

model = load_model('Pedestrian_Detection.h5')
print("Pedestrian detection model loaded successfully!")


def initialize_hog_detector():
   """Initializes the HOG descriptor for pedestrian detection."""
   hog = cv2.HOGDescriptor()
   hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
   return hog

def process_video_frame(frame, hog):
   """Processes a video frame for pedestrian detection."""
   frame = imutils.resize(frame, width=min(400, frame.shape[1]))  # Resize for efficiency
   regions, _ = hog.detectMultiScale(frame, winStride=(4, 4), padding=(4, 4), scale=1.05)
   for (x, y, w, h) in regions:
       cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw bounding boxes
   return frame

def main():
   """Main function for video-based pedestrian detection."""
   hog = initialize_hog_detector()
   cap = cv2.VideoCapture("C:\\Users\\Abhinay\\OneDrive\\Documents\\Pedestrian_Detection\\tourist_crossing_the_street (1080p).mp4")

   while cap.isOpened():
       ret, frame = cap.read()
       if not ret:
           break

       processed_frame = process_video_frame(frame, hog)

       cv2.imshow("Image", processed_frame)
       if cv2.waitKey(25) & 0xFF == ord('q'):
           break

   cap.release()
   cv2.destroyAllWindows()

if __name__ == "__main__":
   main()
