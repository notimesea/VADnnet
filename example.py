from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav
import os

keyword = ["1", "2"]

print keyword.index("1")
print keyword.index("2")
print keyword.index("3")

sph2pipe = "/Users/evgeny/kaldi3/tools/sph2pipe_v2.5/sph2pipe"

path = "/Users/evgeny/timit/TIMIT/TRAIN/DR1/FCJF0/SA1.WAV"

os.system(sph2pipe + " -f wav " + path + " tmp.wav")

window = 0.025
step = 0.01
nfilt = 40
fftsize = 512

(rate,sig) = wav.read("tmp.wav")

os.remove("tmp.wav")

mfcc_feat = mfcc(sig,rate)
fbank_feat = logfbank(sig, rate, window, step, nfilt, fftsize, 0, None, 0)
print sig, rate
print fbank_feat[1:3,:]
