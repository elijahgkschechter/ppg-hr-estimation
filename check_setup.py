import sys
from pathlib import Path
print ("python v:", sys.version.split()[0])
#prints py version


import numpy, scipy, pandas, matplotlib, sklearn
for module in (numpy, scipy, pandas, matplotlib, sklearn):
    print(f"{module.__name__:<12} {module.__version__}")
#imports the version of the pachages to prove install 

project_folder = Path(__file__).resolve().parent
#folder that the script lives in 
data_folder = project_folder.parent / "ppg+dalia" / "PPG_FieldStudy"
#goes into the data folder 
subject_files = sorted(data_folder.glob("S*/S*.pkl"))
#finds a file that matches the subject and pattern 

print("Data folder found:", data_folder.exists())
print("Subject files found:", len(subject_files))