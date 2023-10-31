from scipy.io import loadmat                # Biblioteca para carregar arquivos .mat
import numpy as np                     # Biblioteca do numpy para manipulação de matrizes
import mne
from sklearn.preprocessing import LabelEncoder

file = loadmat("./datasets/beta/S8.mat")

data = file['data'][0][0][0]
# print(data.shape)
labels = file['data'][0][0][1][0][0]

# print("Participante:", labels[0][0])
# print("Idade:", labels[1][0][0])
# print("Gênero:", labels[2][0])
# print("Eletrodos:", len(labels[3]))
# print("Labels:", labels[4][0])
# print("Dados desconhecidos i = 5 - len():", len(labels[5][0]))
# print("Dados desconhecidos i = 6:", labels[6][0][0])
# print("Dados desconhecidos i = 7:", labels[7][0][0])

# chanels = labels[3]
# print("LABELS",labels)
# labels = labels[4][0]

# print(labels)
# data é composto pelos 64 eletrodos, cada um com 650 pontos, 
# e cada um dos 650 pontos contém 4 trials,
# e cada um dos 4 trials contém 40 números (frequências representando as teclas)
# 64 eletrodos
# 750 pontos
# 4 Trials (experimentos)
# 40 teclas/frequências

#Reshape para ir de (64, 750, 4, 40) para (64, 750, 160)
data = np.reshape(data, (64, 750, 160))

# Trocamos a posição dos eixos
data = np.swapaxes(data, 2, 0)
data = np.swapaxes(data, 2, 1)
print("SHAPE",data.shape)

# -------------------------------------- INFO
ch_names = []
# print("LABELS", labels)
event_labels = labels[4][0]
labels = labels[3]
ch_types = []

for label in labels:
    ch_names.append(label[3][0])
    ch_types.append('eeg')

info = mne.create_info(ch_names, sfreq=250, ch_types=ch_types)
print("INFO",info)
#epochs
#criar dicionário das teclas e frequências
event_dict = {
    '.': 0, ',': 1, '<': 2, 'a': 3, 'b': 4,
    'c': 5, 'd': 6, 'e': 7, 'f': 8, 'g': 9,
    'h': 10, 'i': 11, 'j': 12, 'k': 13, '1': 14,
    'm': 15, 'n': 16, 'o': 17, 'p': 18, 'q': 19,
    'r': 20, 's': 21, 't': 22, 'u': 23, 'v': 24,
    'w': 25, 'x': 26, 'y': 27, 'z': 28, '0': 29,
    '1': 30, '2': 31, '3': 32, '4': 33, '5': 34,
    '6': 35, '7': 36, '8': 37, '9': 38, '_': 39
}

# print("-----------------------------------------------------------------------------------")
# print(labels[1].shape)
# print("-----------------------------------------------------------------------------------")
# print(labels)

#criar epoch
le = LabelEncoder()
events = np.column_stack((
    np.array(range(len(event_labels))),
    np.zeros(event_labels.shape[0], dtype=int),
    le.fit_transform(event_labels))
)

mne_data = mne.EpochsArray(data, info, events, event_id=event_dict)
print(mne_data)

#compute_psd
