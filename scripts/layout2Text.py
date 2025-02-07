#Imports
import json
import os
from natsort import natsorted

INPUT_FOLDER = "Layouts"
OUTPUT_FOLDER = "Texts"

def main():
    for book in os.listdir(INPUT_FOLDER):
        if os.path.exists(OUTPUT_FOLDER+"/"+book+".txt"):
            continue
        text = ""
        for page in natsorted(os.listdir(INPUT_FOLDER+"/"+book)):
            page_path = INPUT_FOLDER+"/"+book+"/"+page
            with open(page_path) as jsonObj:
                #Grab saved JSON as dict
                data = json.loads(json.loads(jsonObj.read()))
                for chunk in data['chunkedDocument']['chunks']:
                    text+=chunk['content']
        #Write text to file
        with open(OUTPUT_FOLDER+"/"+book+".txt", 'w', encoding='utf-8') as writer:
            writer.write(text)

if __name__ == "__main__":
    main()