## Dataset Glossary

This folder contains all the dataset files that are used in this project. The files can be categorized into two parts: Malicious and Benign.

### Dataset Information

There are in total 83 different malware families datasets available in the directory. The malicious dataset contains the keyword 'dga' in the name of the file while the name of the benign dataset is `ground_truth_data_processed.csv`. Each file has the following features -

```
Domain (String):	The whole string of the URL
VT_Scan (Int): 		The scan result from the VirusTotal
			0 denotes no record
			1 denotes not malicious
			2 denotes malicious
isNXDomain (Int): 	The scan result from nslookup
			0 denotes NXDomain
			1 denotes not a NXDomain
perNumChars (Int):	This is the percentage ratio of Numerics to Characters in the domain string
VtoC (Int):		This is the percentage ratio of Vowel to Consonant in the domain string
lenDomain (Int):	Length of the domain string
SymToChar (Int):	This is the percentage ratio of Symbols to Characters in the domain string
TLD (String):		Top Level Domain (TLD) of the Domain (Need to OneHotEncode afterwards)
family_id (Int):	The flag for each malware family
Class (Int):		The binary class label for malicious dataset
```

### Master Dataset
There is a csv file, `master_dataset.csv` that has got all the combined records from every file.
