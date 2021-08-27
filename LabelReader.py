# Simple code to grab data from a spreadhseet and use it to poopulate a clinical data dictionary for our brain scans
# Richard Masson
import pandas as pd
import os
import time

# Define file path
root = "C:\\Users\\richa\\Documents\\Uni\\Thesis\\Label Data"
data_loc = "Basic CDR Info.csv" #File to be read in goes here
path = os.path.join(root, data_loc)
print("Reading in data...")
# Create a dataframe and use that to populate a dictionary
tic = time.perf_counter()
df = pd.read_csv(path)
print("Read in data successfully.")
#df = df.head(60) #Only if a subset is needed for testing purposes
scan_dict = df.set_index('ADRC_ADRCCLINICALDATA ID')[['Subject','cdr']].to_dict(orient='index')
# Dictionary refinement go here
d_items = scan_dict.items()
patient_CDR_range = {}
for key, values in d_items:
    sub = values['Subject']
    val = values['cdr']
    if sub not in patient_CDR_range:
        patient_CDR_range[sub] = []
        patient_CDR_range[sub].append(val)
    elif val not in patient_CDR_range[sub]:
        patient_CDR_range[sub].append(val)
print(patient_CDR_range)
toc = round((time.perf_counter()-tic),2)
print("Total time:", toc, "seconds.")