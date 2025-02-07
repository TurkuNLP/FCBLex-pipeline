#Imports
import json
from trankit import trankit2conllu
import os


def main():
    """
    Main function which looks for .json files produced by Trankit and transforms them to CoNLLU files
    """
    #JSON to CoNLLU

    for file in os.listdir("Parsed"):
        #Don't do work if file already exists!
        if os.path.exists("Conllus/"+file.replace(".json", ".conllu")):
            continue
        with open("Parsed/"+file) as json_file:
            #Convert json to dict
            data = json.load(json_file)
            connlu_doc = trankit2conllu(data)
            with open("Conllus/"+file.replace(".json", ".conllu"), "w", encoding="utf-8") as writer:
                writer.write(connlu_doc)
            #doc = Document(data)
            #CoNLL.write_doc2conll(doc, "Parsed/"+file.replace(".json", ".conllu"))

if __name__ == "__main__":
    main()