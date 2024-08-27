import csv


def add_missing_data(results, license_plate_db):
    """
    Function to update the results dictionary by comparing detected license plates with a database
    to determine if the license plate is registered, missing, or an enquiry.

    :param results: Dictionary containing detection results
    :param license_plate_db: Dictionary containing known license plates and their status
                             (e.g., {"ABC123": "registered", "XYZ789": "missing"})
    :return: Updated results dictionary
    """

    for frame_nmr, frame_data in results.items():
        for vehicle_id, vehicle_data in frame_data.items():
            license_plate_text = vehicle_data['license_plate']['text']
            lp_status = license_plate_db.get(license_plate_text, "enquiry")  # Default to "enquiry"

            # Update the license plate data with the status from the database
            results[frame_nmr][vehicle_id]['license_plate']['type'] = lp_status

    return results


def read_license_plate_db(db_path):
    """
    Function to read a CSV file containing known license plates and their statuses.

    :param db_path: Path to the CSV file
    :return: Dictionary of license plates and their statuses
    """
    license_plate_db = {}

    with open(db_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) == 2:
                plate, status = row
                license_plate_db[plate.strip()] = status.strip().lower()

    return license_plate_db
