__author__ = "Md. Ahsan Ayub"
__license__ = "GPL"
__credits__ = ["Ayub, Md. Ahsan", "Smith, Steven", "Tinker, Paul",
               "Siraj, Ambareen"]
__maintainer__ = "Md. Ahsan Ayub"
__email__ = "mayub42@students.tntech.edu"
__status__ = "Prototype"

# Import libraries 
import pandas as pd

#importing the data set
dataset = pd.read_csv('dnschanger_dga_processed.csv')
print(dataset.head())

# Store the number of records
totalRecords = len(dataset.index)
print('Total Records:\t', totalRecords)


# To print unique TLDs in the dataset
#print('TLDs:\t', dataset.TLD.unique())
print('Length of Unique TLDs:\t', len(dataset.TLD.unique()))

# To count the scan results of VirusTotal
noRecords = 0
safeRecords = 0
maliciousRecords = 0

try:
    noRecords = dataset.groupby('VT_scan').count()['domain'][0]
except:
    noRecords = 0
    
try:
    safeRecords = dataset.groupby('VT_scan').count()['domain'][1]
except:
    safeRecords = 0
    
try:
    maliciousRecords = dataset.groupby('VT_scan').count()['domain'][2]
except:
    maliciousRecords = 0

VT_No_Record = (noRecords * 100) / totalRecords
VT_Safe = (safeRecords * 100) / totalRecords
VT_Malicious = (maliciousRecords * 100) / totalRecords

print('VT Scan:\t', VT_No_Record, ' ', VT_Safe, ' ', VT_Malicious)


# To count the number of domains having numerics in it
uniqueCounts = []
uniqueCounts = dataset.groupby('perNumChars').count().index.tolist()

try:
    indexOfZero = uniqueCounts.index(0)
    numNumericRecords = dataset.groupby('perNumChars').count()['domain'][uniqueCounts[indexOfZero]]
    perNonNumericRecords = round((numNumericRecords * 100) / totalRecords)
    print('perNonNumericRecords:\t', perNonNumericRecords)
except:
    print('perNonNumericRecords:\t', 0)
    
# To count the number of live domains
try:
    liveDomain = dataset.groupby('isNXDomain').count()['domain'][1]
    perLiveDomainRecords = round((liveDomain * 100) / totalRecords)
    print('perLiveDomainRecords:\t', perLiveDomainRecords)
except:
    print('perLiveDomainRecords:\t', 0)

# Finding unique records using group by attribute
# complexQueryList = dataset.groupby("TLD").nunique()