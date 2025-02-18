#Imports
import json
import os
from natsort import natsorted

INPUT_FOLDER = "Layouts"
OUTPUT_FOLDER = "UncleanTexts"

def includeIfWantedText(block_text: str) -> str:
    """
    Function for filtering text from the text_blocks of data from GCL Layout Parser
    """
    #These are almost always either titles, sfx or image-texts
    #Ignore purely numerical text blocks, as these are either page numbers or otherwise irrelevant
    if not str(block_text).isnumeric():
        #Ignore blocks which have under three words. These are usually titles or image-texts
        #if str(block_text).count(' ') > 2:
        return block_text
    return ""

def main():
    for book in os.listdir(INPUT_FOLDER):
        if os.path.exists("Texts"+"/"+book+".txt"):
            continue
        helper = 0
        text = ""
        for page in natsorted(os.listdir(INPUT_FOLDER+"/"+book)):
            page_path = INPUT_FOLDER+"/"+book+"/"+page
            with open(page_path) as jsonObj:
                #Grab saved JSON as dict
                data = json.loads(json.loads(jsonObj.read()))
                #For each block on the first level
                for block in data['documentLayout']['blocks']:
                    #Not interested in tables etc.
                    if 'textBlock' in list(block.keys()):
                        #If we're in a title-blocks structure (2nd level)
                        if block['textBlock']['type'] != 'paragraph':
                            for inner_block in block['textBlock']['blocks']:
                                #Still not interested in tables etc.
                                if 'textBlock' in list(inner_block.keys()):
                                    #If we're in heading-2 - blocks structure (3rd level)
                                    if inner_block['textBlock']['type'] != 'paragraph':
                                        for inner_inner_block in inner_block['textBlock']['blocks']:
                                            #Still still not interested in tables etc.
                                            if 'textBlock' in list(inner_inner_block.keys()):
                                                text += includeIfWantedText(inner_inner_block['textBlock']['text'])+'\n'
                                    else:
                                        text += includeIfWantedText(inner_block['textBlock']['text'])+'\n'
                        else:
                            text += includeIfWantedText(block['textBlock']['text'])+'\n'
            text += '\n'+str(helper)+'\n\n'
            helper += 1
                        

        #Write text to file
        with open(OUTPUT_FOLDER+"/"+book+".txt", 'w', encoding='utf-8') as writer:
            writer.write(text)

if __name__ == "__main__":
    main()