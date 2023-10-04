from scipy.io import loadmat                # Biblioteca para carregar arquivos .mat
import numpy as np                     # Biblioteca do numpy para manipulação de matrizes
import mne

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
    '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0,
    '7': 0, '8': 0, '9': 0, '0': 0,
    'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0, 'f': 0,
    'g': 0, 'h': 0, 'i': 0, 'j': 0, 'k': 0, '1': 0,
    'm': 0, 'n': 0, 'o': 0, 'p': 0, 'q': 0, 'r': 0,
    's': 0, 't': 0, 'u': 0, 'v': 0, 'x': 0, 'w': 0,
    'y': 0, 'z': 0, '<': 0, '_': 0, '.': 0, ',': 0
}
#criar epoch
#compute_psd
