#Imports

import pandas as pd


data = pd.read_csv("VRT/no_comments_nlfcl_fi.vrt", sep="\s+", header=None, on_bad_lines="skip")
data.columns = ['text', 'head', 'lemma', 'lemmacomp', 'upos', 'lex', 'morph', 'dependency relation', 'depencdency head', '0', '1', '2', '3', '4']

data=data[data['head'].str.isnumeric() == True]

print(data)

#with pd.ExcelWriter("VRT/vrt_as_xl.xlsx") as writer:
#    data.to_excel(writer)

#with open("/VRT/vrt_data_extracted.txt", "r") as reader:
