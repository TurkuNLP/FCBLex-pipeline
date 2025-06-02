#Imports
import trankit
from trankit import Pipeline
import os
import numpy
import json
from tqdm import tqdm



def main():
    """
    Main function which looks for txt-files and parses them with Trankit
    Outputs .json files with the name ISBN_age_register_parsed.json
    """
    #Setups
    p = Pipeline('finnish')
    #Show progress bar for books
    with tqdm(range(len(os.listdir("Texts"))), desc="Parsing books...") as pbar:
        #For each book
        for file in os.listdir("Texts"):
            #If a folder exists, we assume that that book has already been parsed

            if os.path.exists("Parsed/"+file+"_parsed.json"):
                print("Skipping...")
                pbar.update(1)
                continue
            text=""
            with open("Texts/"+file, "r") as reader:
                text=text+reader.read()
            reader.close()
            #Start with parsing the book
            with open("Parsed/"+file+"_parsed.json", "w", encoding="utf-8") as fp:
                #Parse text if text is not empty
                if len(text)!=0:
                    data = p(text)
                    #Write results to file
                    json.dump(data, fp, ensure_ascii=False)
            pbar.update(1)

if __name__ == "__main__":
    main()