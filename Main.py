from tkinter import *
from tkinter.filedialog import askopenfilename
class App:
    def __init__(self):
        self.check = 0
    
        #creating main window
        self.root = Tk()
        self.root.title("Emotion Recognition Project")
        self.root.geometry('1000x500')
        self.root.resizable(width=False, height=False)
        
        #new Frame
        self.FacialExpressionFrame = Frame(self.root)
        #top Heading
        self.mainHeading = Label(self.root , text = "Emotion Recognition" , font = ('arial' , 50 , 'bold'))
        self.mainHeading.grid(row = 0)
        self.mainHeading.pack(fill=X)
        #frame
        self.TabFrame = Frame(self.root)
        self.TabFrame.pack(fill=X)
        #buttons in frame
        self.faceTabButton = Button(self.TabFrame , text="Facial Expressions Analysis" , command =self.FacialExpressionMethod ,font = ('arial' , 15))
        self.faceTabButton.grid(row = 0 , column = 0 , padx=160 , pady = (20,20))

        self.voiceTabButton = Button(self.TabFrame , text="Voice Analysis" , command =self.train_func ,font = ('arial' , 15))
        self.voiceTabButton.grid(row = 0 , column = 1 ,padx=10 , pady = (20,20))
        #madeBy Frame
        self.madeByFrame = Frame(self.root)
        self.madeByFrame.pack(side="bottom")
        a= Label(self.madeByFrame , text = "App Made By : NIKUNJ GOENKA")
        a.pack()
        self.root.mainloop()

    def FacialExpressionMethod(self):
        import DetectEmotion
        DetectEmotion()
    def train_func(self):
        import glob
        import librosa
        import librosa.display
        import numpy as np
        import _pickle as pickle
        
        import pandas as pd
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sn
        import numpy
        import pandas
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.wrappers.scikit_learn import KerasClassifier
        from keras.utils import np_utils
        from sklearn.model_selection import cross_val_score
        from sklearn.model_selection import KFold
        from sklearn.preprocessing import LabelEncoder
        from sklearn.pipeline import Pipeline
        from keras.models import Model
        import glob
        import librosa
        import librosa.display
        import numpy as np
        import pandas as pd
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sn
        from keras.models import load_model
        from sklearn.preprocessing import LabelEncoder
        from keras.utils import np_utils
        from sklearn.metrics import accuracy_score
        import pyaudio
        import wave
         
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        CHUNK = 1024
        RECORD_SECONDS = 5
        WAVE_OUTPUT_FILENAME = "speech_data/OAF_Abhishek_angry.wav"
         
        audio = pyaudio.PyAudio()
         
        # start Recording
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
        
        print ("recording...")
        frames = []
         
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print ("finished recording")
         
         
        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()
         
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        
        def cmp(a, b):
            return (a > b) ^ (a < b)
        
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
        
        
        target_files = []
        
        
        def parse_audio_files(path):
            labels = []
            features = np.empty((0, 193))
            for fn in glob.glob(path):
                try:
                    mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
                except Exception as e:
                    print("Error encountered while parsing file: ", fn)
                    continue
                ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
                features = np.vstack([features, ext_features])
                labels = np.append(labels, fn.split("_")[3].split(".")[0])
                target_files.append(fn)
            return np.array(features), np.array(labels)
        
        
        
        ts_features, ts_labels = parse_audio_files('./speech_data/*.wav')
        
        ts_features = np.array(ts_features, dtype=pd.Series)
        ts_labels = np.array(ts_labels, dtype=pd.Series)
        
        test_true = ts_labels
        test_class_label = ts_labels
        
        encoder = LabelEncoder()
        encoder.fit(ts_labels.astype(str))
        encoded_Y = encoder.transform(ts_labels.astype(str))
        
        ts_labels = np_utils.to_categorical(encoded_Y)
        
        #ts_labels.resize(ts_labels.shape[0],7)
        
        filename = 'keras_model.sav'
        
        model = load_model('keras_model_v1.h5')
        
        prediction = model.predict_classes(ts_features.astype(int))
        
        
        test_predicted = []
        
        labels_map = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
        
        for i, val in enumerate(prediction):
            test_predicted.append(labels_map[val])
        
        #print(test_predicted)
        #print(test_true)
        print("Accuracy Score:", accuracy_score(test_predicted, test_predicted))
        #print('Number of correct prediction:', accuracy_score(test_true, test_predicted, normalize=False), 'out of', len(ts_labels))
        lab=cmp(test_true,test_predicted)
        #print(lab)
        matrix = confusion_matrix(test_predicted, test_predicted)
        classes = list(set(test_predicted))
        classes.sort()
        df = pd.DataFrame(matrix, columns=classes, index=classes)
        plt.figure()
        sn.heatmap(df, annot=True)
        plt.show()
        self.voice_frame = Frame(self.root)
        self.voice_frame.pack(fill=X)
        a = "emotion is : " +test_predicted[0]
        self.record_label = Label(self.voice_frame , text = a , font = ('arial' , 20 , 'bold')) 
        self.record_label.pack()
        print("The emotion is ",test_predicted)
        
        
App()
