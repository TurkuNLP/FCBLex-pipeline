#Imports
import json
import pyconll
import pandas as pd
import os
import numpy as np
import re
from tqdm import trange, tqdm
import matplotlib
import matplotlib.pyplot as plt
import openpyxl
import seaborn as sns
import math
from scipy.stats import norm


def initBooksFromJsons(json_path: str) -> dict:
    """
    Function which takes in a path to a folder containing .json files produced by Trankit and creates python dicts from them
    :JSON_PATH: path to folder as str
    :return: python dict with form [book_name, pandas.DataFrame]
    """

    books = {}
    #Loading the conllus (jsons) as Dataframes

    for file in os.listdir(json_path):
        #Opening json contents
        with open(json_path+"/"+file) as json_file:
            #Transform into dataframe
            df = pd.read_json(json_file)
            #Append as dict juuuuust in case we need the metadata
            #Clip at 17 as the format for the filenames are standardized
            books[file[:17]] = df
    return books

def initBooksFromConllus(conllu_path: str) -> dict:
    """
    Function which takes in a path to a folder containing conllu files and returns a dict with pd.DataFrames of sentence data
    :conllu_path: path to folder of conllus as str
    :return: dict of form [book_name, pd.DataFrame], df is sentence data of a book
    """

    books = {}
    #Loading the conllus as Dataframes
    for file in os.listdir(conllu_path):
        #Opening conllus contents
        with open(conllu_path+"/"+file) as conllu_file:
            #Transform into dataframe
            df = pd.read_csv(conllu_file, sep="[\t]", header=None, on_bad_lines='skip', engine='python')
            #Set names for columns
            df.columns = ['id', 'text', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc']
            #Append as dict juuuuust in case we need the metadata
            #Clip at 17 as the format for the filenames are standardized
            books[file[:17]] = df
    return books

#Functions which extract data from a dictionary to get data on sentences

#Function that removes PUNCT
def getNoPunct(sentences: dict) -> dict:
    """
    Function which takes in a dict created by getTokenData and spits out a version with the upos tag PUNCT removed from the DataFrames
    Recommend only editing sentence data
    :sentences: dict of form [book_name, pandas.DataFrame]
    :return: dict of form [book_name, pandas.DataFrame] with upos PUNCT removed from df
    """
    no_punct = {}
    #Remove the rows from each dataframe what are classified as PUNCT
    for key in sentences:
        df = sentences[key]
        no_punct[key] = df[df.upos != "PUNCT"]
    return no_punct

#Function that takes in a dictionary of [book_name, conllu_dataframe] and returns a dictionary with [book_name, sentences_dataframe]
def getSentenceData(books: dict) -> dict:
    """
    Function which takes in a dict created by initBooks and spits out a version where DataFrames only contain sentences data (see CoNLLU file format for more info)
    :books: dict of form [book_name, pandas.DataFrame]
    :return: dict of form [book_name, pandas.DataFrame] with only sentence data
    """
    return_dict = {}
    with tqdm(range(len(books.keys())), desc="Extracting sentences...") as pbar:
        #For key-value pair in dict
        for key in books:
            #Init a new array for sentences
            sentence_dfs = []
            df = books[key]
            
            #Only care about the sentences
            for sentence in df['sentences']:
                #Add dfs created from sentences to list
                sentence_dfs.append(pd.DataFrame.from_dict(sentence['tokens']))
            #Map book_name to a dataframe from all its sentences
            sentece_df = pd.concat(sentence_dfs, ignore_index=True)
            return_dict[key]=sentece_df
            #Update pbar
            pbar.update(1)
    return return_dict

#Function which returns a dictionary [book_name, lemma_freq_pivot_table]
def getLemmaFrequencies(sentences: dict) -> dict:
    """
    Function which takes in sentence data and creates a dict with the lemma frequencies of each book
    :sentences: dict of form [book_name, pandas.DataFrame]
    :return: dict of form [book_name, pandas.DataFrame] df contains lemmas and frequencies
    """
    lemma_freqs = {}
    for key in sentences:
        lemma_freqs[key] = getLemmaFreq(sentences[key])
    return lemma_freqs

def getOnlyAlnums(sentences: dict, column: str) -> dict:
    """
    Function which takes in sentence data and cleans punctuation and other non-alnum characters
    :param sentences:dict of form [book_name, pd.DataFrame], df is sentence data
    :param column: name of the column which to clean (recommend 'text' or 'lemma')
    :return: dict of the same form
    """
    clean = {}
    for key in sentences:
        df = sentences[key]
        #Get rid of PUNCT
        no_punct = df[df.upos != "PUNCT"].copy()
        #Make words lowercase
        no_punct[column] = no_punct[column].apply(lambda x: x.lower())
        #Remove non-alnums
        no_punct[column] = no_punct[column].apply(lambda x: ''.join(filter(str.isalnum, x)))
        #Filter rows with nothing in them
        no_punct = no_punct[no_punct.text != '']
        clean[key] = no_punct
    return clean

#Return dataframe with lemmas in descending order (ignoring PUNCT and non alnums)
def getLemmaFreq(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function which takes in a pd.DataFrame (sentence data) and spits out a df which has lemmas and their freqs
    """
    #Return a pivot_table turned DataFrame that counts the occurances of each lemma and sorts them in descending order
    return pd.DataFrame.pivot_table(df, columns='lemma', aggfunc='size').sort_values(ascending=False).reset_index().rename(columns={0: "frequency"})

#Get frequencies of words (not PUNCT, all to lowercase, and removed all non alnums)
def getWordFrequencies(sentences: dict) -> dict:
    """
    Function which takes in a dict of sentence data and spits out a dict with dfs containg word frequencies of the books
    """
    word_freqs = {}
    for key in sentences:
        df = sentences[key]
        #Map book_name to pivot table
        word_freqs[key] = pd.DataFrame.pivot_table(df, columns='text', aggfunc='size').sort_values(ascending=False).reset_index().rename(columns={0: "frequency"})
    return word_freqs


def getTokenAmounts(sentences: dict) -> dict:
    """
    Get amount of tokens in sentences
    """
    word_amounts = {}
    for key in sentences:
        df = sentences[key]
        word_amounts[key] = len(df)
    return word_amounts

#Get PoS frequencies
def getPOSFrequencies(sentences: dict) -> dict:
    """
    Function which gets the POS frequencies of sentences of books
    """
    pos_freqs = {}

    for key in sentences:
        #Map book_name to pivot table
        pos_freqs[key] = pd.DataFrame.pivot_table(sentences[key], columns='upos', aggfunc='size').sort_values(ascending=False).reset_index().rename(columns={0: "frequency"})

    return pos_freqs


#Functions to get metrics from sentences

#Function the get the average length of the unique lemmas in the sentenes
def getAvgLen(data: dict, column: str) -> dict:
    """
    Get the average length of either words or lemmas from sentence data
    :data: dict of form [book_name, pd.DataFrame], df should contain sentence data
    :column: name of the CoNLLU column for which to calculate the average. Recommend either 'text' for words and 'lemma' for lemmas
    :return: dict of [book_name, avg_len], where avg_len is float
    """
    avg_lens = {}
    for key in data:
        i = 1
        total_len = 0
        df = data[key]
        #For each lemma count the length and add one to counter
        for lemma in df[column]:
            #Only care about strings
            if type(lemma) is str:
                total_len += len(lemma)
                i += 1
        #If no lemmas were found (should never happen but just in case), we make the avg_len be 0
        if i==1:
            avg_lens[key] = 0
        else:
            #Map book_name to avg lemma length
            avg_lens[key] = total_len/(i-1.0)
    return avg_lens

def getAvgSentenceLens(books: dict) -> dict:
    """
    Functon for gettign the average length of sentences in each book
    :param books: dict of form [id, pd.DataFrame] like in the other methods
    :return: dict of form [id, double], where the double is the average sentence length of the corresponding book
    """
    lens = {}
    for key in books:
        df = books[key]
        help1 = len(df)
        nums = df.id.value_counts()
        num_of_sents = nums.iloc[0]
        lens[key] = (help1/num_of_sents)
    return lens

#Function to calculate DP (deviation of proportions) of all the words in the corpus
def getDP(v: dict, f_df: pd.DataFrame, s: dict) -> pd.DataFrame:
    """
    Function which calculates the dispersion (DP) based on the formula by Greis
    :v: dict of form [book_name, pd.DataFrame], df has frequencies per book
    :f_df: pd.DataFrame that includes the total frequencies of words/lemmas in the whole corpus
    :s: dict of form [book_name, ratio], where ratio is how much of the whole corpus a book takes
    :return: pd.DataFrame that has the columns 'text', 'DP', 'DP_norm'
    """
    #First get the minimum s
    min_s = 1
    for key in s:
        if s[key] < min_s:
            min_s = s[key]
    #For corpus parts that are length 1
    if min_s == 1:
        min_s = 0

    v_series = {}
    #Transform v into more usable form
    for key in v:
        v_df = v[key]
        ser = v_df[v_df.columns[1]]
        ser.index = v_df[v_df.columns[0]]
        v_series[key] = ser
    
    texts = []
    DP = []
    DP_norm = []
    with tqdm(range(len(f_df.index)), desc="DP calculations") as pbar:

        #Loop through every single word in the corpus
        for k in range(len(f_df.index)):
            #Get the freq of the word in the whole corpus
            word = f_df.iloc[k, 0]
            f = f_df.iloc[k, 1]
            abs_sum = 0
            #For each document in the corpus
            for key in v_series:
                #Freq of word in document. Set to 0 if not found
                v_i = 0
                try:
                    v_i = v_series[key].loc[word]*1.0
                except:
                    v_i = 0.0
                #Comparative size of document to whole corpus
                s_i = s[key]
                #Calculate the abs_sum used in calculating DP as written by Gries [2020]
                abs_sum += abs(((v_i)/f)-s_i)
            #Append word to list
            texts.append(word)
            #Calculate and append DP
            dp = 0.5*abs_sum
            DP.append(dp)
            #Append DP_norm to list (alltho with how many documents we have, the normalization doesn't work very well at all)
            DP_norm.append(dp/(1-min_s))
            #Update pbar
            pbar.update(1)
    return pd.DataFrame({'text': texts, 'DP': DP, 'DP_norm': DP_norm})

#Function to get contextual diversity
def getCD(v: dict):
    """
    Function which gets the contextual diversity of words/lemmas based on frequency data
    """
    #Get number of books
    books_num = len(v.keys())
    word_series = []
    #For each dataframe attached to a book, look for a frequency list and gather all the words in a list
    for key in v:
        v_df = v[key]
        word_series.append(v_df[v_df.columns[0]])
    #Add all words to a new dataframe
    series = pd.concat(word_series, ignore_index=True)
    #Create pivot table to count in how many books does a word appear in
    CD_raw = series.value_counts()
    #Return Contextual Diversity by dividing the number of appearances by the total number of books
    return CD_raw.apply(lambda x: x/books_num)

#Functions for getting values for different variables used in metrics


def getL(word_amounts: dict) -> int:
    """
    Function for getting the total length of the corpus in terms of the number of words
    """
    l = 0
    for key in word_amounts:
        l += word_amounts[key]
    return l

def getS(word_amounts: dict, l: int) -> dict:
    """
    Function for getting how big each part is in relation to the total size of the corpus
    """
    s = {}
    for key in word_amounts:
        s[key] = (word_amounts[key]*1.0)/l
    return s


def combineFrequencies(freq_data: dict) -> pd.DataFrame:
    """
    Get the total frequencies of passed freq_data in the corpus
    """
    dfs = []
    #Add all dataframes to list
    for key in freq_data:
        dfs.append(freq_data[key])
    #Concat all dataframes together
    df = pd.concat(dfs, ignore_index=True)
    #Return a dataframe containing text in one column and total freq in collection in the other
    return df.groupby(df.columns[0])['frequency'].sum().reset_index()


#Functions to do with sub-corpora


def getSubCorp(corp: dict, num: int) -> dict:
    """
    Simple function to get sub_corpora from the whole package based on the target age group
    Naming conventions are ISBN_age-group_register, where age-group is an int [1,3]
    """
    sub_corp = {}
    for key in corp:
        if key.find('_'+str(num)+'_') != -1:
            sub_corp[key] = corp[key]
    return sub_corp


def combineSubCorpDicts(corps: list) -> dict:
    """
    Combine a list of sub-corp dicts into one dict
    """
    whole = corps[0].copy()
    for i in range(1, len(corps)):
        whole.update(corps[i])
    return whole

def combineSubCorpsData(corps: list):
    """
    Takes in a list of dataframes (or series) and combines them together
    """
    dfs = []
    for df in corps:
        dfs.append(df)
    combined = pd.concat(dfs)
    if type(combined) is pd.DataFrame:
        return combined.groupby(combined.columns[0])[combined.columns[1]].sum().reset_index()
    else:
        return combined.groupby(level=0).sum().reset_index()
    

def getTypeTokenRatios(v: dict, word_amounts: dict) -> pd.Series:
    """
    Function which gets the type-token ratios of each book that's in the corpus
    :param v:frequency data per book
    :param word_amounts:token amounts per book
    :return: pd.Series with book names being indexes and ttr being values 
    """
    names = []
    ttrs = []
    for key in word_amounts:
        v_df = v[key]
        #Get the number of unique entities in freq data
        types = len(v_df)
        #Get the number of token in book
        tokens = word_amounts[key]
        #Add ttr to lis
        ttrs.append(types/tokens)
        #Add key to list
        names.append(key)
    return pd.Series(ttrs, names)

def getZipfValues(l: int, f: pd.DataFrame) -> pd.Series:
    """
    Function for calculating the Zipf values of words/lemmas in a corpus
    Zipf = ( (raw_freq + 1) / (Tokens per million + Types per million) )+3.0
    :param l: total length of corpus (token amount)
    :param f: df containing frequency data of words/lemmas for the corpus
    :return: pd.DataFrame, where indexes are words/lemmas and values the Zipf values
    """
    indexes = list(f[f.columns[0]])
    types_per_mil = len(indexes)/1000000
    tokens_per_mil = l/1000000
    zipfs = f[f.columns[1]]+1
    zipfs = zipfs / (tokens_per_mil + types_per_mil)
    zipfs = np.log10(zipfs)
    zipfs = zipfs + 3.0
    zipfs.index = indexes
    return zipfs

def cleanLemmas(sentences: dict) -> dict:
    """
    Function which takes in sentence data and cleans rows based on a number of filters on lemma forms to guarantee better quality data
    Does get rid of PUNCT
    :param sentences:dict of form [book_name, pd.DataFrame], df is sentence data
    :return: dict of the same form, but better data
    """
    clean = {}
    for key in sentences:
        df = sentences[key]
        no_punct = df.copy()

        #Make words lowercase
        no_punct['lemma'] = no_punct['lemma'].apply(lambda x: str(x).lower())
        #First mask
        #Remove lemmas which are not alnum or have '-' but no weird chars at start or end, length >1, has no ' ', and has no ','
        m = no_punct.lemma.apply(lambda x: (x.isalnum() 
                                            or (not x.isalnum() and '-' in x and x[0].isalnum() and x[len(x)-1].isalnum())
                                            or (not x.isalnum() and '#' in x and x[0].isalnum() and x[len(x)-1].isalnum())
                                            and len(x)>1 
                                            and not ' ' in x
                                            and not ',' in x))
        filtered = no_punct[m]
        #Second mask
        #Remove lemmas that have the same character more than thrice consecutively at the start (Finnish doesn't work like this)
        m_2 = no_punct.lemma.apply(lambda x: conseqChars(x)
                                   and not (x.isnumeric() and len(x)>4)
                                   )
        filtered_2 = filtered[m_2] 
        clean[key] = filtered_2
    return clean

def conseqChars(x: str):
    if len(x)>2:
        return not x[0]==x[1]==x[2]
    else:
        return True
    
def cleanWordBeginnings(books: dict) -> dict:
    """
    Function for cleaning non-alnum characters from the beginning of words in sentence data
    :param books: dict of the sentence data of books
    :return: dictionary, where the dataframes have been cleaned
    """
    #Clean words
    clean = {}
    for key in books:
        df = books[key].copy()
        df['text'] = df['text'].apply(lambda x: delNonAlnumStart(x))
        clean[key] = df
    return clean

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

def ignoreOtherAlphabets(sentences: dict) -> dict:
    """
    Function which takes in sentence data and cleans rows based on if words contain characters not in the Finnish alphabe (e.g. cyrillic)
    :param sentences:dict of form [book_name, pd.DataFrame], df is sentence data
    :return: dict of the same form, but better data
    """
    clean = {}
    for key in sentences:
        df = sentences[key]
        no_punct = df.copy()

        #Make words lowercase
        no_punct['text'] = no_punct['text'].apply(lambda x: str(x).lower())
        #First mask
        #Remove words which are not in the Finnish alphabet
        m = no_punct.text.apply(lambda x: (
                len(x.encode("ascii", "ignore")) == len(x)
                or x.find('ä') != -1
                or x.find('ö') != -1 
                or x.find('å') != -1
            )
        )
        filtered = no_punct[m]
        clean[key] = filtered
    return clean

def getSharedWords(wordFrequencies1: dict, wordFrequencies2: dict) -> pd.DataFrame:
    """
    Gives a pd.DataFrame object where there are two columns: first contains those words/lemmas which are shared and the second their combined frequencies
    """
    sub1 = combineFrequencies(wordFrequencies1)
    sub2 = combineFrequencies(wordFrequencies2)

    shared = pd.concat([sub1, sub2], ignore_index=True)
    mask = shared.duplicated(keep=False)
    shared = shared[mask]
    return shared.groupby(shared.columns[0])['frequency'].sum().reset_index()

#Writing all data into one big xlsx-file
def writeDataToXlsx(name, f_words, f_lemmas, pos_freqs, lemma_DP, word_DP, lemma_CD, word_CD, avg_uniq_lens_df, avg_lens_df):
    """
    Write all wanted data to an xlsx file for testing purposes
    """
    with pd.ExcelWriter("Data/"+name+".xlsx") as writer:
        f_words.to_excel(writer, sheet_name="Word frequencies")
        f_lemmas.to_excel(writer, sheet_name="Lemma frequencies")
        pos_freqs.to_excel(writer, sheet_name="POS frequencies")
        lemma_DP.to_excel(writer, sheet_name="Lemma dispersion")
        word_DP.to_excel(writer, sheet_name="Word dispersion")
        lemma_CD.to_excel(writer, sheet_name="Lemma contextual diversity")
        word_CD.to_excel(writer, sheet_name="Word contextual diversity")
        avg_uniq_lens_df.to_excel(writer, sheet_name="Average unique lengths by book")
        avg_lens_df.to_excel(writer, sheet_name="Average lengths by book")