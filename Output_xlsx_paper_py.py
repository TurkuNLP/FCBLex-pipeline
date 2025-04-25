#Imports
from scripts import bookdatafunctions as bdf
import pandas as pd

corpus = bdf.initBooksFromConllus('Conllus')

corpus = bdf.mapGroup2Age(corpus, "ISBN_MAPS/ISBN2AGE.xlsx")

#Sort the books so that we get groupings by age group
key1 = []
key2 = []
key3 = []
for key in corpus.keys():
    age = int(bdf.findAgeFromID(key))
    if age<=8:
        key1.append(key)
    elif 8<age<13:
        key2.append(key)
    else:
        key3.append(key)
sorted_keys = key1+key2+key3

sorted_keys = list(corpus.keys())
sorted_keys.sort(key=lambda x:int(bdf.findAgeFromID(x)))

from scripts import bookdatafunctions as bdf
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#Constants
JSON_PATH = "Parsed"
CONLLU_PATH = "Conllus"
ISBN2AGE_PATH = "ISBN_MAPS/ISBN2AGE.xlsx"



#books = bdf.initBooksFromJsons(JSON_PATH)
sentences = bdf.mapGroup2Age(bdf.cleanWordBeginnings(bdf.initBooksFromConllus(CONLLU_PATH)), ISBN2AGE_PATH)
#Move to working with just sentence data
#Whole corpus
def formatDataForPaperOutput(corpus: dict[str,pd.DataFrame]):
    ages = sorted(bdf.getAvailableAges(corpus))

    ready_dfs_ages = {}
    ready_dfs_groups = {}
    ready_dfs_whole = {}

    #Subcorpora based on the target age groups
    sub_corpora = []
    #Combine books aged 15 and up into one sub-corpus as there are very few entries in 16,17,18
    over_15 = []
    for i in ages:
        if i<15:
            sub_corpora.append(bdf.cleanWordBeginnings(bdf.getDistinctSubCorp(corpus, i)))
        else:
            over_15.append(bdf.cleanWordBeginnings(bdf.getDistinctSubCorp(corpus, i)))
    #Sort the aged 15 and over sub-corpora from lowest age to highest
    over_15.sort(key=lambda x:int(bdf.findAgeFromID(list(x.keys())[0])))
    #Combine 15+ aged books into one sub-corpus
    sub_corpora.append(bdf.combineSubCorpDicts(over_15))
    #Sort the sub-corpora from lowest age to highest
    sub_corpora.sort(key=lambda x:int(bdf.findAgeFromID(list(x.keys())[0])))
    #Keep track of when words first appear in terms of intended reading age
    word_age_appearances = {}
    #First sort out the subcorpora
    for sub_corp in sub_corpora:
        combined_data = pd.concat(sub_corp.values()).reset_index()
        filtered_data = combined_data[['text','lemma','upos']]
        filtered_data = filtered_data.drop_duplicates(['text','lemma','upos'], ignore_index=True)
        #Add word-pos frequencies
        v_words_pos = bdf.getColumnFrequencies(sub_corp, ['text','upos'])
        word_pos_freqs = bdf.combineFrequencies(v_words_pos)
        filtered_data['Word-POS Frequency'] = [word_pos_freqs[x[0]][x[1]] for x in filtered_data[['text','upos']].to_numpy(dtype='str')]
        #Add word frequencies
        v_words = bdf.getColumnFrequencies(sub_corp, ['text'])
        word_freqs = bdf.combineFrequencies(v_words)
        filtered_data['Word Frequency'] = [word_freqs[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add word zipf-values
        l = bdf.getL(bdf.getTokenAmounts(sub_corp))
        word_zipfs = bdf.getZipfValues(l, word_freqs)
        filtered_data['Word Zipf'] = [word_zipfs[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add word DP
        word_DP = bdf.getDP(v_words, word_freqs, bdf.getS(bdf.getTokenAmounts(sub_corp), l))[0]
        filtered_data['Word DP'] = [word_DP[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add word CD
        word_CD = bdf.getCD(v_words)
        filtered_data['Word CD'] = [word_CD[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add lemma frequencies
        v_lemmas = bdf.getColumnFrequencies(sub_corp, ['lemma'])
        lemma_freqs = bdf.combineFrequencies(v_lemmas)
        filtered_data['Lemma Frequency'] = [lemma_freqs[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add word zipf-values
        lemma_zipfs = bdf.getZipfValues(l, lemma_freqs)
        filtered_data['Lemma Zipf'] = [lemma_zipfs[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add word DP
        lemma_DP = bdf.getDP(v_lemmas, lemma_freqs, bdf.getS(bdf.getTokenAmounts(sub_corp), l))[0]
        filtered_data['Lemma DP'] = [lemma_DP[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add word CD
        lemma_CD = bdf.getCD(v_lemmas)
        filtered_data['Lemma CD'] = [lemma_CD[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add taivutusperhe size
        tv_sizes = bdf.getTaivutusperheSize(sub_corp)
        filtered_data['Lemma inflection family size'] = [tv_sizes[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        key = bdf.findAgeFromID(list(sub_corp.keys())[0])
        #Slow but steady way of adding words and first appearance ages...
        for w in word_freqs.index:
            word_age_appearances.setdefault(w[0],key)
        #Add first appearance
        filtered_data['First Age Encountered'] = [word_age_appearances[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add to dictionary
        ready_dfs_ages[key] = filtered_data.sort_values('text')
    
    #Define age group sub-corpora

    
    #Generate correct keys/ids
    group_1 = [5,6,7,8]
    group_2 = [9,10,11,12]
    group_3 = ages[ages.index(13):]
    #Distinct subcorpora
    sub_corp_1= bdf.combineSubCorpDicts([bdf.getDistinctSubCorp(corpus, x) for x in group_1])
    sub_corp_2= bdf.combineSubCorpDicts([bdf.getDistinctSubCorp(corpus, x) for x in group_2])
    sub_corp_3= bdf.combineSubCorpDicts([bdf.getDistinctSubCorp(corpus, x) for x in group_3])
    sub_corps = dict(zip(['7-8','9-12','13+'],[sub_corp_1, sub_corp_2, sub_corp_3]))

    for s in sub_corps:
        sub_corp = sub_corps[s]
        combined_data = pd.concat(sub_corp.values()).reset_index()
        filtered_data = combined_data[['text','lemma','upos']]
        filtered_data = filtered_data.drop_duplicates(['text','lemma','upos'], ignore_index=True)
        #Add word-pos frequencies
        v_words_pos = bdf.getColumnFrequencies(sub_corp, ['text','upos'])
        word_pos_freqs = bdf.combineFrequencies(v_words_pos)
        filtered_data['Word-POS Frequency'] = [word_pos_freqs[x[0]][x[1]] for x in filtered_data[['text','upos']].to_numpy(dtype='str')]
        #Add word frequencies
        v_words = bdf.getColumnFrequencies(sub_corp, ['text'])
        word_freqs = bdf.combineFrequencies(v_words)
        filtered_data['Word Frequency'] = [word_freqs[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add word zipf-values
        l = bdf.getL(bdf.getTokenAmounts(sub_corp))
        word_zipfs = bdf.getZipfValues(l, word_freqs)
        filtered_data['Word Zipf'] = [word_zipfs[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add word DP
        word_DP = bdf.getDP(v_words, word_freqs, bdf.getS(bdf.getTokenAmounts(sub_corp), l))[0]
        filtered_data['Word DP'] = [word_DP[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add word CD
        word_CD = bdf.getCD(v_words)
        filtered_data['Word CD'] = [word_CD[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add lemma frequencies
        v_lemmas = bdf.getColumnFrequencies(sub_corp, ['lemma'])
        lemma_freqs = bdf.combineFrequencies(v_lemmas)
        filtered_data['Lemma Frequency'] = [lemma_freqs[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add word zipf-values
        lemma_zipfs = bdf.getZipfValues(l, lemma_freqs)
        filtered_data['Lemma Zipf'] = [lemma_zipfs[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add word DP
        lemma_DP = bdf.getDP(v_lemmas, lemma_freqs, bdf.getS(bdf.getTokenAmounts(sub_corp), l))[0]
        filtered_data['Lemma DP'] = [lemma_DP[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add word CD
        lemma_CD = bdf.getCD(v_lemmas)
        filtered_data['Lemma CD'] = [lemma_CD[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        #Add taivutusperhe size
        tv_sizes = bdf.getTaivutusperheSize(sub_corp)
        filtered_data['Lemma inflection family size'] = [tv_sizes[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
        key = s
        #Add first appearance
        filtered_data['First Age Encountered'] = [word_age_appearances[x] for x in filtered_data['text'].to_numpy(dtype='str')]
        #Add to dictionary
        ready_dfs_groups[key] = filtered_data.sort_values('text')

    #Work with the whole corpus
    combined_data = pd.concat(corpus.values()).reset_index()
    filtered_data = combined_data[['text','lemma','upos']]
    filtered_data = filtered_data.drop_duplicates(['text','lemma','upos'], ignore_index=True)
    #Add word-pos frequencies
    v_words_pos = bdf.getColumnFrequencies(corpus, ['text','upos'])
    word_pos_freqs = bdf.combineFrequencies(v_words_pos)
    filtered_data['Word-POS Frequency'] = [word_pos_freqs[x[0]][x[1]] for x in filtered_data[['text','upos']].to_numpy(dtype='str')]
    #Add word frequencies
    v_words = bdf.getColumnFrequencies(corpus, ['text'])
    word_freqs = bdf.combineFrequencies(v_words)
    filtered_data['Word Frequency'] = [word_freqs[x] for x in filtered_data['text'].to_numpy(dtype='str')]
    #Add word zipf-values
    l = bdf.getL(bdf.getTokenAmounts(corpus))
    word_zipfs = bdf.getZipfValues(l, word_freqs)
    filtered_data['Word Zipf'] = [word_zipfs[x] for x in filtered_data['text'].to_numpy(dtype='str')]
    #Add word DP
    word_DP = bdf.getDP(v_words, word_freqs, bdf.getS(bdf.getTokenAmounts(corpus), l))[0]
    filtered_data['Word DP'] = [word_DP[x] for x in filtered_data['text'].to_numpy(dtype='str')]
    #Add word CD
    word_CD = bdf.getCD(v_words)
    filtered_data['Word CD'] = [word_CD[x] for x in filtered_data['text'].to_numpy(dtype='str')]
    #Add lemma frequencies
    v_lemmas = bdf.getColumnFrequencies(corpus, ['lemma'])
    lemma_freqs = bdf.combineFrequencies(v_lemmas)
    filtered_data['Lemma Frequency'] = [lemma_freqs[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
    #Add word zipf-values
    lemma_zipfs = bdf.getZipfValues(l, lemma_freqs)
    filtered_data['Lemma Zipf'] = [lemma_zipfs[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
    #Add word DP
    lemma_DP = bdf.getDP(v_lemmas, lemma_freqs, bdf.getS(bdf.getTokenAmounts(corpus), l))[0]
    filtered_data['Lemma DP'] = [lemma_DP[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
    #Add word CD
    lemma_CD = bdf.getCD(v_lemmas)
    filtered_data['Lemma CD'] = [lemma_CD[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
    #Add taivutusperhe size
    tv_sizes = bdf.getTaivutusperheSize(corpus)
    filtered_data['Lemma inflection family size'] = [tv_sizes[x] for x in filtered_data['lemma'].to_numpy(dtype='str')]
    #Add first appearance
    filtered_data['First Age Encountered'] = [word_age_appearances[x] for x in filtered_data['text'].to_numpy(dtype='str')]
    #Add to dictionary
    ready_dfs_whole['Whole'] = filtered_data.sort_values('text')

    return ready_dfs_ages, ready_dfs_groups, ready_dfs_whole
        
        
        
        
def writePaperOutputAges(ready_dfs: dict[str:pd.DataFrame], name: str):
    with pd.ExcelWriter("Data/FCBLex_data_output_"+name+".xlsx") as writer:
        for df in ready_dfs:
            ready_dfs[df].to_excel(writer, sheet_name=df, index=False)
            print(df+" done!")

dfs_ages, dfs_groups, dfs_whole = formatDataForPaperOutput(sentences)
writePaperOutputAges(dfs_ages, 'ages')
print("Ages done!")
writePaperOutputAges(dfs_groups, 'groups')
print("Groups done!")
writePaperOutputAges(dfs_whole, 'whole')
print("Whole done!")

print("Done done!")