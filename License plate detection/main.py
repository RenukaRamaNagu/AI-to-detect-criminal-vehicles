import cv2
import winsound  # For Windows sound alert
from ultralytics import YOLO
import numpy as np
import csv
import os

# Function to trigger and stop the alert sound
def alert_sound():
    winsound.Beep(1000, 500)  # Beep at 1000 Hz for 500 ms

# Function to write results to CSV
def write_csv(data, output_csv='detections.csv'):
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, 'a', newline='') as csvfile:
        fieldnames = ['Frame', 'Class', 'License Plate', 'Confidence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(data)

# Load the trained YOLO model
license_plate_detector = YOLO('runs/detect/train/weights/best.pt')

# List of image paths
image_paths = ['02a3ba4c3886fe9a_jpg.rf.b1adb7907ad0902bc8d611c1b10ef941.jpg']

# Initialize list to store results
csv_data = []

# Process each image
for frame_nmr, image_path in enumerate(image_paths):
    frame = cv2.imread(image_path)
    if frame is not None:
        # Detect license plates
        detections = license_plate_detector(frame)[0]

        # Prepare an array to store the detected bounding boxes
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            detections_.append([x1, y1, x2, y2, score, class_id])

            #  bounding box and label on the image
            label = license_plate_detector.names[int(class_id)]
            if label == "Registered-license-plate":
                color = (0, 255, 0)  # Green for registered
            elif label == "Missing-license-plate":
                color = (0, 165, 255)  # Orange for missing
            elif label == "Enquiry-license-plate":
                color = (0, 0, 255)  # Red for enquiry
            else:
                color = (255, 255, 255)  # White for other/unknown

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f'{label} {score:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Extract license plate text
            license_plate_text = f"{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}"

            # Append data to CSV list
            csv_data.append({
                'Frame': frame_nmr,
                'Class': label,
                'License Plate': license_plate_text,
                'Confidence': f'{score:.2f}'
            })

            # Trigger the beep if "Enquiry-license-plate" is detected
            if label == "Enquiry-license-plate":
                print("Detected an Enquiry_license_plate. Triggering beep.")
                cv2.imshow(f'Frame {frame_nmr}', frame)  # Show the frame before beeping
                cv2.waitKey(1)  # Allow the frame to update
                alert_sound()

        # Display the frame with the bounding boxes
        cv2.imshow(f'Frame {frame_nmr}', frame)

        # Save the result image with bounding boxes
        output_image_path = f'output_frame_{frame_nmr}.jpg'
        cv2.imwrite(output_image_path, frame)
        print(f'Saved frame to {output_image_path}')

# Write the collected data to CSV
write_csv(csv_data, 'license_plate_detections007.csv')

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Processing complete. CSV file created as 'license_plate_detections007.csv'.")
