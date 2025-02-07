#Imports
import re
import os
import pandas as pd
import bookdatafunctions as bdf

def delNonAlnumStart(x: str) -> str:
    '''
    Function for deleting non-alnum sequences of words from Conllu-files
    :param x: string that is at least 2 characters long
    :return: the same string, but with non-alnum characters removed from the start until the first alnum-character
    '''
    if not x[0].isalnum() and len(x)>1:
        ind = 0
        for i in range(len(x)):
            if x[i].isalnum():
                ind=i
                break
        return x[ind:]
    return x

#def cyrillicOrArabic(x: str) -> str:


#Clean words from Conllus
def main():
    #Load the Conllus
    books = bdf.initBooksFromConllus("Conllus")

    #Clean words
    clean = {}
    for key in books:
        df = books[key].copy()
        df['text'] = df['text'].apply(lambda x: delNonAlnumStart(x))
        clean[key] = df

    #Write dfs to conllus
    for key in clean:
        with open(str(key)+".conllu", 'w', encoding="utf-8") as writer:
            pd.DataFrame(clean[key]).to_csv(writer, sep="\t", index=False)

if __name__ == "__main__":
    main()