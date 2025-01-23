import bookdatafunctions as bdf

books = bdf.initBooksFromConllus("Conllus")

lemma_freqs = bdf.getLemmaFrequencies(books)

f_lemma = bdf.getTotal(lemma_freqs)

print(f_lemma)