#Imports
import pandas as pd
import bookdatafunctions as bdf

#def cyrillicOrArabic(x: str) -> str:


#Clean words from Conllus
def main():
    #Load the Conllus
    books = bdf.initBooksFromConllus("Conllus")

    #Clean words
    clean = {}
    for key in books:
        df = books[key].copy()
        df['text'] = df['text'].apply(lambda x: bdf.delNonAlnumStart(x))
        clean[key] = df

    #Write dfs to conllus
    for key in clean:
        with open(str(key)+".conllu", 'w', encoding="utf-8") as writer:
            pd.DataFrame(clean[key]).to_csv(writer, sep="\t", index=False)

if __name__ == "__main__":
    main()