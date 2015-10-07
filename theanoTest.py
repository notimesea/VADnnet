import pickle
import theano
import theano.tensor as T
from pylearn2.models import mlp
from pylearn2.training_algorithms import sgd
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.costs.cost import Cost
from pylearn2.costs.cost import DefaultDataSpecsMixin
import numpy as np
from random import randint
import os
import matplotlib.pyplot as plt
from features import logfbank


class XOR(DenseDesignMatrix):
    def __init__(self):
        self.class_names = ['0', '1']
        X = [[randint(0, 1), randint(0, 1)] for _ in range(1000)]
        y = []
        for a, b in X:
            if a + b == 1:
                y.append([0, 1])
            else:
                y.append([1, 0])
        X = np.array(X)
        y = np.array(y)
        super(XOR, self).__init__(X=X, y=y)


class NegativeLogLikelihoodCost(DefaultDataSpecsMixin, Cost):
    supervised = True

    def expr(self, model, data, **kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)

        inputs, targets = data

        outputs = model.fprop(inputs)

        loss = -(targets * T.log(outputs)).sum(axis=1)

        return loss.mean()

left_context = 10
right_context = 5
keywords = ["she", "had"]

def getDataForFrames(data, X, y):
    fbanks = data[0]
    class_id = data[1]
    # print class_id
    feat_cnt = len(fbanks[0])
    for i in xrange(len(fbanks)):
        features = np.empty(0)
        #print features
        for j in xrange(-left_context, right_context):
            if i + j < 0 or i + j >= len(fbanks):
                features = np.append(features, np.zeros(feat_cnt))
            else:
                features = np.append(features, fbanks[i + j])
        #print features
        result = np.zeros(len(keywords) + 1)
        result[class_id] = 1.0

        X.append(features)
        y.append(result)


def getWindowedFeats(fbanks):
    X = []
    feat_cnt = len(fbanks[0])
    for i in xrange(len(fbanks)):
        features = np.empty(0)
        for j in xrange(-left_context, right_context):
            if i + j < 0 or i + j >= len(fbanks):
                features = np.append(features, np.zeros(feat_cnt))
            else:
                features = np.append(features, fbanks[i + j])
        X.append(features)
    return X

def getDataFromPath(path, maxFiles = 1e9):
    X = []
    y = []
    for file in os.listdir(path):
        maxFiles -= 1
        if maxFiles < 0:
            break
        data = np.load(path + "/" + file)
        getDataForFrames(data, X, y)

    X = np.array(X)
    y = np.array(y)

    return X, y

class SHEHAD(DenseDesignMatrix):
    feat_cnt = 0
    def __init__(self):
        self.class_names = ["she", "had", "filler"]
        X, y = getDataFromPath("/Users/evgeny/data/TRAIN")
        # print X.shape
        self.feat_cnt = len(X[0])
        super(SHEHAD, self).__init__(X=X, y=y)

def test(model):
    X, y = getDataFromPath("/Users/evgeny/data/TEST")

    confusion = np.zeros([3, 3])

    ypred = np.log(model.fprop(theano.shared(np.array(X), name='inputs')).eval())

    cnt = np.zeros(3)

    for a, b in zip(ypred, y):
        pos = np.argmax(b)
        for i in xrange(3):
            confusion[pos][i] += a[i]
        cnt[pos] += 1

    for i in xrange(3):
        for j in xrange(3):
            confusion[i][j] /= cnt[i]

    print confusion

# create hidden layer with 2 nodes, init weights in range -0.1 to 0.1 and add
# a bias with value 1

rng = 0.01

modelName = "2x128relu50epochs.mdl"
debug = False


if debug or not os.path.exists(modelName):
    ds = SHEHAD()
    hidden_layer = mlp.RectifiedLinear(layer_name='hidden', dim=128, irange=rng, init_bias=rng)
    hidden_layer2 = mlp.RectifiedLinear(layer_name='hidden2', dim=128, irange=rng, init_bias=rng)
    hidden_layer3 = mlp.RectifiedLinear(layer_name='hidden3', dim=128, irange=rng, init_bias=rng)
    # create Softmax output layer
    output_layer = mlp.Softmax(3, 'output', irange=.1)
    # create Stochastic Gradient Descent trainer that runs for 400 epochs
    cost = NegativeLogLikelihoodCost()
    trainer = sgd.SGD(learning_rate=.05, cost=cost,  batch_size=64, termination_criterion=EpochCounter(50))
    layers = [hidden_layer, hidden_layer2,output_layer]
    # create neural net that takes two inputs
    ann = mlp.MLP(layers, nvis=ds.feat_cnt)
    trainer.setup(ann, ds)
    print trainer.cost
    # train neural net until the termination criterion is true
    while True:
        trainer.train(dataset=ds)
        ann.monitor.report_epoch()
        ann.monitor()
        if not trainer.continue_learning(ann):
            break
    with open(modelName, 'wb') as f:
        if not debug:
            pickle.dump(ann, f)
else:
    with open(modelName) as f:
        ann = pickle.load(f)

#test(ann)

window = 0.025
step = 0.01
nfilt = 40
fftsize = 512

def extractLogFBank(rate, sig):
    feats = logfbank(sig, rate, window, step, nfilt, fftsize, 0, None, 0)
    return feats

#sph2pipe = "/Users/evgeny/kaldi3/tools/sph2pipe_v2.5/sph2pipe"

#os.system(sph2pipe + " -f wav " + "SA1.WAV" + " tmp.wav")

def computeFile(model, path):
    # import extractFeats
    import scipy.io.wavfile as wav

    (rate, sig) = wav.read(path)

    fbanks = extractLogFBank(rate, sig)

    X = getWindowedFeats(fbanks)

    ypred = np.log(model.fprop(theano.shared(np.array(X), name='inputs')).eval())

    ypred = np.transpose(ypred)

    plt.plot(ypred[0])
    plt.show()
    plt.plot(ypred[1])
    plt.show()
    plt.plot(ypred[2])
    plt.show()

computeFile(ann, "tmp2.wav")