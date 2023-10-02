from scipy.io import loadmat                # Biblioteca para carregar arquivos .mat
import numpy as np                          # Biblioteca do numpy para manipulação de matrizes

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
labels = labels[4][0]

# print(labels)
# data é composto pelos 64 eletrodos, cada um com 650 pontos, 
# e cada um dos 650 pontos contém 4 trials,
# e cada um dos 4 trials contém 40 números (frequências representando as teclas)
# 64 eletrodos
# 750 pontos
# 4 Trials (experimentos)
# 40 teclas/frequências

# print(chanels[0][3]) chanel name

#Reshape para ir de (64, 750, 4, 40) para (64, 750, 160)
data = np.reshape(data, (64, 750, 160))

# Trocamos a posição dos eixos
data = np.swapaxes(data, 2, 0)
data = np.swapaxes(data, 2, 1)
print(data.shape)