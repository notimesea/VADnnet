__author__ = 'evgeny'

from features import logfbank
from collections import defaultdict
import scipy.io.wavfile as wav
import os
import numpy as np

sph2pipe = "/Users/evgeny/kaldi3/tools/sph2pipe_v2.5/sph2pipe"

window = 0.025
step = 0.01
nfilt = 40
fftsize = 512

left_context = 6
right_context = 2

def getAudio(path):
    os.system(sph2pipe + " -f wav " + path + " tmp.wav")
    (rate, sig) = wav.read("tmp.wav")
    os.remove("tmp.wav")
    return (rate, sig)

def extractLogFBank(rate, sig):
    feats = logfbank(sig, rate, window, step, nfilt, fftsize, 0, None, 0)
    return feats


keywords = ["she", "had"]

src = "/Users/evgeny/timit/TIMIT"
dst = "/Users/evgeny/data"

count = {"TEST": 10000, "TRAIN": 100000}


for type in ["TEST", "TRAIN"]:
    dstt = dst + "/" + type
    srct = src + "/" + type
    if not os.path.exists(dstt):
        os.mkdir(dstt)

    mx = count[type]
    item_id = 0

    used = set()

    wordCount = defaultdict(int)

    for root, dirs, filenames in os.walk(srct):
        for f in filenames:
            if f.endswith(".WRD"):
                id = f[:-4]
                rate, sig = getAudio(root + "/" + id + ".WAV")
                file = open(root + "/" + f)
                for line in file:
                    [start, end, word] = line.split()
                    start = int(start)
                    end = int(end) + 1
                    sig_seg = sig[start:end]
                    #print root, id, start, end
                    fbanks = extractLogFBank(rate, sig_seg)
                    class_id = keywords.index(word) if word in keywords else len(keywords)
                    wordCount[class_id] += 1
                    np.save(dst + "/" + type + "/" + str(item_id), [fbanks, class_id])
                    item_id += 1

                file.close()

    print item_id
    for w in sorted(wordCount, key=wordCount.get, reverse=True):
        print w, wordCount[w]
