import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import tensorflow as tf

def extract_features(file_path, max_pad_len):
    audio, sample_rate = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_fft=400, hop_length=160, n_mfcc=100)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=400, hop_length=160, n_mels=128)

    # Mel-Spectrogram을 dB 단위로 변환
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # 패딩 처리
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width < 0:
        mfccs = mfccs[:, :max_pad_len]
    else:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

    pad_width = max_pad_len - mel_spectrogram_db.shape[1]
    if pad_width < 0:
        mel_spectrogram_db = mel_spectrogram_db[:, :max_pad_len]
    else:
        mel_spectrogram_db = np.pad(mel_spectrogram_db, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # MFCC 데이터의 차원을 변경 (height, width, 1)
    mfccs = np.expand_dims(mfccs, axis=-1)

    # MFCC 데이터를 3채널로 변경 (height, width, 3)
    mfccs = np.concatenate([mfccs, mfccs, mfccs], axis=-1)

    return mfccs, mel_spectrogram_db

def get_max_length(file_list, folder):
    max_length = 0
    for file in file_list:
        if file.endswith('.wav'):
            audio, sample_rate = librosa.load(os.path.join(folder, file), sr=16000)
            length = audio.shape[0]
            if length > max_length:
                max_length = length
    return max_length

ai_folder = 'C:/Users/seungyeon0510/Desktop/PBL/2024/vc_ko_500_part12'
human_folder = 'C:/Users/seungyeon0510/Desktop/PBL/2024/human_ko_500(from ai_hub)'

ai_files = os.listdir(ai_folder)
human_files = os.listdir(human_folder)

max_length_ai = get_max_length(ai_files, ai_folder)
max_length_human = get_max_length(human_files, human_folder)

max_length = max(max_length_ai, max_length_human)
max_pad_len = max_length // 160 + 1  # n_fft=400, hop_length=160 을 고려하여 max_pad_len 계산

mfccs_data = []  
mel_spectrogram_data = []  
labels = []  

for file in ai_files:
    if file.endswith('.wav'):
        mfccs, mel_spectrogram = extract_features(os.path.join(ai_folder, file), max_pad_len)
        mfccs_data.append(mfccs)
        mel_spectrogram_data.append(mel_spectrogram)
        labels.append(0)  # AI 음성 레이블

for file in human_files:
    if file.endswith('.wav'):
        mfccs, mel_spectrogram = extract_features(os.path.join(human_folder, file), max_pad_len)
        mfccs_data.append(mfccs)
        mel_spectrogram_data.append(mel_spectrogram)
        labels.append(1)  # Human 음성 레이블
        
# 데이터를 훈련 세트와 테스트 세트로 분할
mfccs_train, mfccs_test, mel_spectrogram_train, mel_spectrogram_test, labels_train, labels_test = train_test_split(mfccs_data, mel_spectrogram_data, labels, test_size=0.2, random_state=42)

mfccs_train = np.array(mfccs_train)  # 추가된 코드
mfccs_test = np.array(mfccs_test)  # 추가된 코드
mel_spectrogram_train = np.array(mel_spectrogram_train)  # 추가된 코드
mel_spectrogram_test = np.array(mel_spectrogram_test)  # 추가된 코드

labels_train = to_categorical(labels_train, num_classes=2)
labels_test = to_categorical(labels_test, num_classes=2)

# 모델 구성 및 학습
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(100, max_pad_len, 3))
for layer in base_model.layers:
    layer.trainable = False
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
vgg19_output = Dense(2, activation='softmax')(x)
vgg19_model = Model(inputs=base_model.input, outputs=vgg19_output)

input_shape = (128, max_pad_len)
input_layer = Input(shape=input_shape)

bilstm_model = tf.keras.models.Sequential([
    input_layer,
    Bidirectional(LSTM(512, return_sequences=False)),
    Dropout(0.8),
    Dense(2, activation='sigmoid')
])

ensemble_inputs = [vgg19_model.input, bilstm_model.input]
ensemble_outputs = tf.keras.layers.average([vgg19_model.output, bilstm_model.output])
ensemble_model = Model(inputs=ensemble_inputs, outputs=ensemble_outputs)

ensemble_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

ensemble_model.fit([mfccs_train, mel_spectrogram_train], labels_train, epochs=10, batch_size=8)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return precision, recall, f1, accuracy

# CNN 모델 성능평가
cnn_precision, cnn_recall, cnn_f1, cnn_accuracy = evaluate_model(vgg19_model, mfccs_test, labels_test)
print('CNN - Precision: {:.2f}, Recall: {:.2f}, F1-score: {:.4f}, Accuracy: {:.2f}'.format(cnn_precision, cnn_recall, cnn_f1, cnn_accuracy))

# BiLSTM 모델 성능평가
bilstm_precision, bilstm_recall, bilstm_f1, bilstm_accuracy = evaluate_model(bilstm_model, mel_spectrogram_test, labels_test)
print('BiLSTM - Precision: {:.2f}, Recall: {:.2f}, F1-score: {:.4f}, Accuracy: {:.2f}'.format(bilstm_precision, bilstm_recall, bilstm_f1, bilstm_accuracy))

# Ensemble 모델 성능평가
ensemble_precision, ensemble_recall, ensemble_f1, ensemble_accuracy = evaluate_model(ensemble_model, [mfccs_test, mel_spectrogram_test], labels_test)
print('Ensemble - Precision: {:.2f}, Recall: {:.2f}, F1-score: {:.4f}, Accuracy: {:.2f}'.format(ensemble_precision, ensemble_recall, ensemble_f1, ensemble_accuracy))
