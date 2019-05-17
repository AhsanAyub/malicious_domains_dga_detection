import os
import glob
import pandas as pd

os.chdir("<dir_name>")

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

all_filenames = sorted(all_filenames)

# Combine all files in the list
master_data = pd.concat([pd.read_csv(f) for f in all_filenames])

# Export to csv
master_data.to_csv( "master_dataset.csv", index=False, encoding='utf-8-sig')