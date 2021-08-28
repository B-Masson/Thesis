# Function file for reading the values out of BIDS folders without using mne-bids
# Richard Masson
import os
import os.path as op
import json

def getDict(root):
        bids_root = op.join(root, "BIDS")
        dir = os.listdir(bids_root)
        bids = "null"
        if len(dir) != 1:
            print("Why the hell are there multiple files")
        elif ".json" in dir[0]:
                bids = op.join(bids_root, dir[0])
        f = open(bids, 'r')
        data = json.load(f)
        f.close()
        return data