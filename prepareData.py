__author__ = 'evgeny'

from features import logfbank
import scipy.io.wavfile as wav
import os


sph2pipe = "/Users/evgeny/kaldi3/tools/sph2pipe_v2.5/sph2pipe"

window = 0.025
step = 0.01
nfilt = 40
fftsize = 512

def extractLogFBank(path):
    os.system(sph2pipe + " -f wav " + path + " tmp.wav")
    (rate, sig) = wav.read("tmp.wav")
    feats = logfbank(sig, rate, window, step, nfilt, fftsize, 0, None, 0)
    os.remove("tmp.wav")
    return feats


path = "/Users/evgeny/timit"

for root, dirs, filenames in os.walk(path):
    for f in filenames:
        if f.endswith(".WRD"):
            id = f[:-4]
            print id
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