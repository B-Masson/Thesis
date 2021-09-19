# Simple code to grab data from a spreadhseet and use it to poopulate a clinical data dictionary for our brain scans
# Richard Masson
import pandas as pd
import os
import time

# Define file path
def loadCDR(data_loc = "Basic CDR Info.csv", size = 0): #File to be read in goes here
    root = "C:\\Users\\richa\\Documents\\Uni\\Thesis\\Label Data"
    path = os.path.join(root, data_loc)
    # Create a dataframe and use that to populate a dictionary
    tic = time.perf_counter()
    df = pd.read_csv(path)
    print("Read in data successfully.")
    if size != 0:
        df = df.head(size) #Only if a subset is needed for testing purposes
    scan_dict = df.set_index('ADRC_ADRCCLINICALDATA ID')[['Subject','cdr']].to_dict(orient='index')
    # Dictionary refinement go here
    d_items = scan_dict.items()
    patient_CDR_range = {}
    for key, values in d_items: #Create a dictionary showing the range of CDR values for each given patient
        id_seg = key.split('_')
        sub = values['Subject']
        val = values['cdr']
        day = int(id_seg[2][1:])
        if sub not in patient_CDR_range:
            patient_CDR_range[sub] = []
        patient_CDR_range[sub].append([day, val])
        '''
        if sub not in patient_CDR_range:
            patient_CDR_range[sub] = []
            patient_CDR_range[sub].append(val)
        elif val not in patient_CDR_range[sub]:
            patient_CDR_range[sub].append(val)
        '''
    toc = round((time.perf_counter()-tic),2)
    #print("Total time:", toc, "seconds.") # For testing
    print("Meta data set up.")
    return scan_dict, patient_CDR_range

#scan, pat = loadCDR()
#print(pat['OAS30001'])