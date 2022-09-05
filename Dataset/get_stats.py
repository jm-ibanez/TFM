import os
import numpy as np
from astropy.table import Table, join
from matplotlib import pyplot as plt
import pandas as pd


datapath = "/opt/TFM/DATASETS/GZ2/"
dataset_files = "/opt/TFM/DATASETS/new_subset/my_files.txt"


dataset_files = "/opt/TFM/new_subset_full.txt"
gz2_file = '/opt/TFM/DATASETS/GZ2/zoo2MainSpecz.csv'

gz2_table = pd.read_csv(gz2_file)
# gz2_table = pd.read_csv('http://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz', compression='gzip')

#gz2_table = Table.read(os.path.join(datapath, "gz2_hart16.fits.gz"))

gz2_table['simple_class'] = gz2_table['gz2class'].apply(lambda x: x[0])

print("TYPES = \n", gz2_table['simple_class'].value_counts())


# Other way to count types values
i_E = gz2_table[gz2_table['simple_class'] == 'E']
i_S = gz2_table[gz2_table['simple_class'] == 'S']

print("E = {}, S={}".format(i_S.shape, i_E.shape))


# Read new_subset_DS
my_files = list(np.loadtxt(dataset_files, dtype="str"))

spiral = 0
ellipical = 0
unknown = 0
for f in my_files:
    id = int(os.path.basename(f).split(".")[0].split("_")[1])
    #print("ID", id)
    gal = gz2_table[gz2_table['dr7objid'] == id]
    #print("VALUES -->", gal['simple_class'].values)

    #if gz2_table[gz2_table['dr7objid'] == id]['simple_class'].values[0] == 'E':
    #    ellipical +=1
    #else:
    #    spiral +=1

    if len(gal) > 0:
        if gal['simple_class'].values[0] == 'E':
            ellipical += 1
        elif gal['simple_class'].values[0] == 'S':
            spiral += 1
        else:
            print("File with Star or Artifacts--> ", f)
            unknown +=1
    else:
        print("dr7objid not found: ", f)


print("\nType E=%d , S=%d, U=%d"%(ellipical, spiral, unknown))
