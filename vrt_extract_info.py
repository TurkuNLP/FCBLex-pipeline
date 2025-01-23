#Imports

import pandas as pd


data = pd.read_csv("VRT/vrt_data_extracted.txt", sep="\t", header=None, on_bad_lines="skip")
data.columns = ['text', 'head', 'lemma', 'upos']

with pd.ExcelWriter("VRT/vrt_as_xl.xlsx") as writer:
    data.to_excel(writer)

#with open("/VRT/vrt_data_extracted.txt", "r") as reader:
