import glob
import librosa
import librosa.display
import numpy as np
import numpy
import _pickle as pickle
from sklearn import svm
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.layers import Dropout
from keras.utils import plot_model
from IPython.display import SVG
from matplotlib import pyplot as plt

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                              sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz


def parse_audio_files(path):
    features, labels = np.empty((0, 193)), np.empty(0)
    labels = []
    for fn in glob.glob(path):
        try:
            mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
        except Exception as e:
            print("Error encountered while parsing file: ", fn)
            continue
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        features = np.vstack([features, ext_features])
        labels = np.append(labels, fn.split("_")[3].split(".")[0])

    return np.array(features), np.array(labels)
tr_features, tr_labels = parse_audio_files('./speech_data1/*.wav')


tr_features = np.array(tr_features, dtype=pd.Series)
tr_labels = np.array(tr_labels, dtype=pd.Series)

X = tr_features.astype(int)
Y = tr_labels.astype(str)
X.shape, Y.shape
seed = 7
numpy.random.seed(seed)

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

dummy_y = np_utils.to_categorical(encoded_Y)
dummy_y
def baseline_model():
    deep_model = Sequential()
    deep_model.add(Dense(100, input_dim=193, activation="relu", kernel_initializer="uniform"))
    deep_model.add(Dropout(0.5))
    deep_model.add(Dense(50, activation="relu", kernel_initializer="uniform"))
    deep_model.add(Dropout(0.5))
    deep_model.add(Dense(20, activation="relu", kernel_initializer="uniform"))
    deep_model.add(Dropout(0.5))
    deep_model.add(Dense(7, activation="softmax", kernel_initializer="uniform"))

    deep_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return deep_model


epoches = 1000
batch_size = 30
verbose = 1

model = baseline_model()
result = model.fit(X, dummy_y, validation_split=0.1, batch_size=batch_size, epochs=epoches, verbose=verbose)

filename = 'keras_model.h5'

model.save(filename)

print('Model Saved..')

plt.plot(result.history['acc'])
plt.plot(result.history['val_acc'])
plt.title('keras model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('keras model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()def parse_audio(path):
    labels = []
    labels1 = []
    features = np.empty((0, 193))
    for fn in glob.glob(path):
        try:
            mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
        except Exception as e:
            print("Error encountered while parsing file: ", fn)
            continue
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        features = np.vstack([features, ext_features])
        labels = np.append(labels,fn)
        labels1 = np.append(labels1, fn.split("_")[3].split(".")[0])
        #print (labels)
        target_files.append(fn)
    return labels,labels1
ts_labels,ts_labels1 = parse_audio('./speech_data/*.wav')
#print(test_predicted)
#print(test_true)
print ("Surprise:")
if 'ps' in test_predicted:
    i=0
    for test in test_predicted:
        if((test=='ps')&((ts_labels[i].split("_")[3].split(".")[0])=='ps')):
            print(ts_labels[i])
        i=i+1
else:
    print("0")
print ("Neutral:")
if 'neutral' in test_predicted:
    i=0
    for test in test_predicted:
        if((test=='neutral')&((ts_labels[i].split("_")[3].split(".")[0])=='neutral')):
            print(ts_labels[i])
        i=i+1
else:
    print("0")
print ("Sad:")
if 'sad' in test_predicted:
    i=0
    for test in test_predicted:
        if((test=='sad')&((ts_labels[i].split("_")[3].split(".")[0])=='sad')):
            print(ts_labels[i])
        i=i+1
else:
    print("0")
        
print ("Happy:")
if 'happy' in test_predicted:
    i=0
    for test in test_predicted:
        if((test=='happy')&((ts_labels[i].split("_")[3].split(".")[0])=='happy')):
            print(ts_labels[i])
        i=i+1
else:
    print("0")
print ("Disgust:")
if 'disgust' in test_predicted:
    i=0
    for test in test_predicted:
        if((test=='disgust')&((ts_labels[i].split("_")[3].split(".")[0])=='disgust')):
            print(ts_labels[i])
        i=i+1
else:
    print("0")
print ("Fear:")
if 'fear' in test_predicted:
    i=0
    for test in test_predicted:
        if((test=='fear')&((ts_labels[i].split("_")[3].split(".")[0])=='fear')):
            print(ts_labels[i])
        i=i+1
else:
    print("0")
print ("Angry")
if 'angry' in test_predicted:
    i=0
    for test in test_predicted:
        if((test=='angry')&((ts_labels[i].split("_")[3].split(".")[0])=='angry')):
            print(ts_labels[i])
        i=i+1
else:
    print("0")