import os
import glob
import pandas as pd

os.chdir("/Users/ahsanayub/Documents/Reseach/Scholarly Papers/Malware Generated Domains/malicious_domains_dga/Dataset")

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

# Combine all files in the list
master_data = pd.concat([pd.read_csv(f) for f in all_filenames])

# Export to csv
master_data.to_csv( "master_dataset.csv", index=False, encoding='utf-8-sig')