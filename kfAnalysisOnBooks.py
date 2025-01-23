import bookdatafunctions as bdf

books = bdf.initBooksFromConllus("Conllus")

sub_2 = bdf.getSubCorp(books, 2)
sub_3 = bdf.getSubCorp(books, 3)

lemma_freqs = bdf.getLemmaFrequencies(books)

word_freqs_2 = bdf.getWordFrequencies(sub_2)
word_freqs_3 = bdf.getWordFrequencies(sub_3)

word_amounts_2 = bdf.getWordAmounts(sub_2)
word_amounts_3 = bdf.getWordAmounts(sub_3)

ttrs_2 = bdf.getTypeTokenRatios(word_freqs_2, word_amounts_2)
ttrs_3 = bdf.getTypeTokenRatios(word_freqs_3, word_amounts_3)

print(ttrs_2.mean())
print(ttrs_3.mean())