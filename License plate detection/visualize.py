import cv2

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

    for vehicle_id, details in results.get(frame_nmr, {}).items():
        # Print details for debugging
        print(f"Details for vehicle_id {vehicle_id}: {details}")

        # Get bounding box and details for the vehicle
        vehicle_bbox = details.get('vehicle', {}).get('bbox', [])
        lp_bbox = details.get('license_plate', {}).get('bbox', [])
        lp_text = details.get('license_plate', {}).get('text', '')
        lp_type = details.get('license_plate', {}).get('type', '')
        lp_score = details.get('license_plate', {}).get('bbox_score', 0.0)

        if len(vehicle_bbox) < 4 or len(lp_bbox) < 4:
            print(f"Invalid bounding box data for vehicle_id {vehicle_id}")
            continue

        # Define colors based on the license plate type
        if lp_type == "registered":
            color = (0, 255, 0)  # Green for registered license plates
        elif lp_type == "missing":
            color = (0, 165, 255)  # Orange for missing license plates
        else:
            color = (0, 0, 255)  # Red for enquiry license plates

        # Draw bounding box around the vehicle (optional)
        cv2.rectangle(frame, (int(vehicle_bbox[0]), int(vehicle_bbox[1])),
                      (int(vehicle_bbox[2]), int(vehicle_bbox[3])),
                      (255, 255, 255), 2)  # White for vehicle bounding box

        # Draw bounding box around the license plate
        cv2.rectangle(frame, (int(lp_bbox[0]), int(lp_bbox[1])),
                      (int(lp_bbox[2]), int(lp_bbox[3])),
                      color, 2)

        # Display license plate text, type, and confidence score
        label = f"{lp_type.capitalize()} ({lp_score:.1f})"
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top_left = (int(lp_bbox[0]), int(lp_bbox[1]) - label_size[1] - 10)
        bottom_right = (int(lp_bbox[0]) + label_size[0], int(lp_bbox[1]))

        # Background rectangle for text
        cv2.rectangle(frame, top_left, bottom_right, color, cv2.FILLED)

        # Display text
        cv2.putText(frame, label, (int(lp_bbox[0]), int(lp_bbox[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame

