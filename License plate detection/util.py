import cv2
import numpy as np
import pytesseract
import csv
import os

# Function to read the license plate text using OCR
def read_license_plate(license_plate_crop):
    # Configure Tesseract's OCR options for better accuracy
    custom_config = r'--oem 3 --psm 8'  # OEM 3: Default, PSM 8: Treat the image as a single word
    license_plate_text = pytesseract.image_to_string(license_plate_crop, config=custom_config)

    # Clean the OCR result (remove unwanted characters, etc.)
    license_plate_text = ''.join(e for e in license_plate_text if e.isalnum())

    # Calculate a confidence score (this is a simple approximation)
    text_score = np.mean([ord(char) for char in license_plate_text]) / 255 if license_plate_text else 0

    return license_plate_text, text_score

# Function to write the results to a CSV file
def write_csv(data, output_csv='detections.csv'):
    """
    Write detection results to a CSV file.
    :param data: List of dictionaries containing detection results.
    :param output_csv: Path to the CSV file to write.
    """
    file_exists = os.path.isfile(output_csv)
    with open(output_csv, 'a', newline='') as csvfile:
        fieldnames = ['Frame', 'Class', 'License Plate', 'Confidence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(data)

# Function to visualize detection results on the given frame
def visualize_results(frame, results, frame_nmr):
    """
    Visualize detection results on the given frame.
    :param frame: The image frame to visualize.
    :param results: Dictionary containing detection results.
    :param frame_nmr: Frame number to identify which results to display.
    :return: The frame with visualized results.
    """
    if frame_nmr not in results:
        print(f"No results for frame {frame_nmr}")
        return frame

    for vehicle_key, details in results.get(frame_nmr, {}).items():
        # Extract integer vehicle id from key
        try:
            vehicle_id = int(vehicle_key.split('_')[1])
        except ValueError:
            print(f"Invalid vehicle_key: {vehicle_key}")
            continue

        # Check for existence of keys
        if 'license_plate' not in details:
            print(f"Key 'license_plate' not found in details for vehicle_key {vehicle_key}")
            continue

        # Get details for the license plate
        lp_bbox = details['license_plate'].get('bbox', [])
        lp_type = details['license_plate'].get('type', '')
        lp_score = details['license_plate'].get('bbox_score', 0.0)

        # Extract license plate crop and read text using OCR
        if len(lp_bbox) == 4:
            x1, y1, x2, y2 = map(int, lp_bbox)
            lp_crop = frame[y1:y2, x1:x2]
            lp_text, text_score = read_license_plate(lp_crop)
        else:
            lp_text, text_score = '', 0.0

        # Define colors based on the license plate type
        if lp_type == "registered":
            color = (0, 255, 0)  # Green for registered license plates
        elif lp_type == "missing":
            color = (0, 165, 255)  # Orange for missing license plates
        else:
            color = (0, 0, 255)  # Red for enquiry license plates

        # Display license plate type and confidence score
        label = f"{lp_text} ({lp_type.capitalize()}, {lp_score:.1f}, {text_score:.1f})"
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top_left = (10, 30 + (30 * vehicle_id))  # Position text with offset
        bottom_right = (top_left[0] + label_size[0], top_left[1] + label_size[1] + base_line)

        # Background rectangle for text
        cv2.rectangle(frame, top_left, bottom_right, color, cv2.FILLED)

        # Display text
        cv2.putText(frame, label, (top_left[0], top_left[1] + label_size[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame

# Function to display the frame
def display_frame(frame, window_name="Detection Results"):
    """
    Display the frame with visualization results.
    :param frame: The image frame to display.
    :param window_name: The name of the window displaying the frame.
    """
    cv2.imshow(window_name, frame)
    cv2.waitKey(0)  # Press any key to close the window
    cv2.destroyAllWindows()

# Function to save the frame with results
def save_frame(frame, output_path):
    """
    Save the frame with visualization results.
    :param frame: The image frame to save.
    :param output_path: The path where the frame will be saved.
    """
    cv2.imwrite(output_path, frame)
    print(f'Saved frame to {output_path}')

