import bookdatafunctions as bdf
import pandas as pd
import seaborn as sns
import matplotlib

books = bdf.initBooksFromConllus("Conllus")

sub_2 = bdf.getSubCorp(books, 2)
sub_3 = bdf.getSubCorp(books, 3)

lemma_freqs = bdf.getLemmaFrequencies(books)

word_freqs_2 = bdf.getWordFrequencies(sub_2)
word_freqs_3 = bdf.getWordFrequencies(sub_3)

word_amounts_2 = bdf.getTokenAmounts(sub_2)
word_amounts_3 = bdf.getTokenAmounts(sub_3)

ttrs_2 = bdf.getTypeTokenRatios(word_freqs_2, word_amounts_2)
ttrs_3 = bdf.getTypeTokenRatios(word_freqs_3, word_amounts_3)

print(ttrs_2.mean())
print(ttrs_3.mean())

word_amounts = bdf.getTokenAmounts(books)

l = bdf.getL(word_amounts)

f_lemma = bdf.getTotal(lemma_freqs)

lemma_zipfs = bdf.getZipfValues(l, f_lemma)

print(lemma_zipfs)

ax = lemma_zipfs.plot.hist()