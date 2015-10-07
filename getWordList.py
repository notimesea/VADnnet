__author__ = 'evgeny'

import os
from collections import defaultdict

src = "/Users/evgeny/timit"

wordCount = defaultdict(int)

for root, dirs, filenames in os.walk(src):
    for f in filenames:
        if f.endswith(".WRD"):
            sent = []
            file = open(root + "/" + f)
            for line in file:
                words = line.split()
                for word in words:
                    if word.isalpha():
                        wordCount[word] += 1
                        sent.append(word)
            file.close()

for w in sorted(wordCount, key=wordCount.get, reverse=True):
  print w, wordCount[w]