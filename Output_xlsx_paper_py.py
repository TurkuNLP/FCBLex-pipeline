#Imports
from scripts import bookdatafunctions as bdf
import pandas as pd

corpus = bdf.initBooksFromConllus('Conllus')

corpus = bdf.mapGroup2Age(corpus, "ISBN_MAPS/ISBN2AGE.xlsx")

#Sort the books so that we get groupings by age group
key1 = []
key2 = []
key3 = []
for key in corpus.keys():
    age = int(bdf.findAgeFromID(key))
    if age<=8:
        key1.append(key)
    elif 8<age<13:
        key2.append(key)
    else:
        key3.append(key)
sorted_keys = key1+key2+key3

sorted_keys = list(corpus.keys())
sorted_keys.sort(key=lambda x:int(bdf.findAgeFromID(x)))

from scripts import bookdatafunctions as bdf
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#Constants
JSON_PATH = "Parsed"
CONLLU_PATH = "Conllus"
ISBN2AGE_PATH = "ISBN_MAPS/ISBN2AGE.xlsx"


#Init corpus from CoNLLU files
corpus = bdf.mapGroup2Age(bdf.cleanWordBeginnings(bdf.initBooksFromConllus(CONLLU_PATH)), ISBN2AGE_PATH)       

#Use the monster function (see bookdatafunctions.py) to get correctly formatted DataFrames
dfs_ages, dfs_groups, dfs_whole = bdf.formatDataForPaperOutput(corpus)
#Output xlsx-files
bdf.writePaperOutputAges(dfs_ages, 'ages')
print("Ages done!")
bdf.writePaperOutputAges(dfs_groups, 'groups')
print("Groups done!")
bdf.writePaperOutputAges(dfs_whole, 'whole')
print("Whole done!")

print("Done done!")