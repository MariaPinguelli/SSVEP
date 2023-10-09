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
    '1': 14, '2': 14.2, '3': 14.4, '4': 14.6, '5': 14.8, '6': 15,
    '7': 15.2, '8': 15.4, '9': 15.6, '0': 13.8,
    'a': 8.6, 'b': 8.8, 'c': 9, 'd': 9.2, 'e': 9.4, 'f': 9.6,
    'g': 9.8, 'h': 10, 'i': 10.2, 'j': 10.4, 'k': 10.6, '1': 10.8,
    'm': 11, 'n': 11.2, 'o': 11.4, 'p': 11.6, 'q': 11.8, 'r': 12,
    's': 12.2, 't': 12.4, 'u': 12.6, 'v': 12.8, 'x': 13.2, 'w': 13,
    'y': 13.4, 'z': 13.6, '<': 8.4, '_': 15.8, '.': 8, ',': 8.2
}

# print("-----------------------------------------------------------------------------------")
# print(labels[1].shape)
# print("-----------------------------------------------------------------------------------")
# print(labels)

#criar epoch
le = LabelEncoder()
events = np.column_stack((
    np.array(range(len(labels))),
    np.zeros(labels.shape[0], dtype=float),
    le.fit_transform(event_labels))
)

mne_data = mne.EpochsArray(data, info, events, event_id=event_dict)
print(mne_data)

#compute_psd
