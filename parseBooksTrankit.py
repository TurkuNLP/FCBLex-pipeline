#Imports
import trankit
from trankit import Pipeline
import os
import numpy
import json
from natsort import natsorted



def main():
    """
    Main function which looks for txt-files produced by Tesseract and parses them with Trankit
    Outputs .json files with the name ISBN_age_register_parsed.json
    """
    #Setups
    p = Pipeline('finnish')

    #For each book
    for folder in os.listdir("Texts"):
        #If a folder exists, we assume that that book has already been parsed

        if os.path.exists("Parsed/"+folder+"_parsed.json"):
            continue
        #Combine all pages together into a book
        text=""
        #For each page
        for page in natsorted(os.listdir("Texts/"+folder)):
            with open("Texts/"+folder+"/"+page) as reader:
                text=text+reader.read()
            reader.close()
        #Start with parsing the book
        with open("Parsed/"+folder+"_parsed.json", "w", encoding="utf-8") as fp:
            #Parse text if text is not empty
            if len(text)!=0:
                data = p(text)
                #Write results to file
                json.dump(data, fp, ensure_ascii=False)

if __name__ == "__main__":
    main()