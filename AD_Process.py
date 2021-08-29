# Application class for pre-processing all the data needed to train a model for Alzheimer's Disease prediction
# Richard Masson
import LabelReader as lr
import NIFTI_Engine as ne
import numpy as np

# Fetch all our seperated data
print("Starting up")
x_arr, scan_meta = ne.extractArrays('anat3')
clinic_sessions, cdr_meta = lr.loadCDR(size=50)
print('*'*10)

# Ascertain lengths and shapes
print("Scan array length:", len(x_arr), "| Scan meta length:", len(scan_meta))
print("Clinical sessions length:", len(clinic_sessions), "| Patient CDR meta length:", len(cdr_meta))
print(scan_meta)
print('*'*10)

# Generate some cdr y labels for each scan
# Current plan: Use the cdr from the closest clinical entry by time (won't suffice in the longterm but it will do for now)
y_arr = []
for scan in scan_meta:
    scan_day = scan['day']
    print("Patient", scan['ID'], "scan on day:", scan_day)
    scan_cdr = -1
    cdr_day = -1
    min = 1000
    for x in cdr_meta[scan['ID']]:
        diff = abs(scan_day-x[0])
        if diff < min:
            min = diff
            scan_cdr = x[1]
            cdr_day = x[0]
    print("Assigned cdr", scan_cdr, "- from day", cdr_day)
    y_arr.append(scan_cdr)
print('*'*10)
print(y_arr)

# Feed into some training model (TO DO)
# trainModel(x_arr, y_arr)