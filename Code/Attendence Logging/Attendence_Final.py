from collections import Counter
from datetime import datetime
import csv

# Read recognized names
with open("recognized_names01.txt", "r") as f:
    names = [line.strip() for line in f if line.strip() != "Unknown"]

# Generate attendance for each recognized person
if names:
    print("[INFO] Generating attendance report...")
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Iterate over recognized names and add them to attendance CSV
    with open("attendance_report.csv", "a", newline="") as file:
        writer = csv.writer(file)
        for name in set(names):  # Add each person only once
            writer.writerow([name, date, time])

    print(f"[INFO] Attendance report updated at {time} on {date}.")
else:
    print("[INFO] No valid recognized names found.")
