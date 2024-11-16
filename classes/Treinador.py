import os
import cv2
import numpy as np
from itertools import cycle

# Função para processar o rótulo em formato de coordenadas de contorno
def convert_to_yolo_format_from_contour(points, width, height):
    """
    Converte uma lista de pontos para o formato YOLO (classe, x_centro, y_centro, largura, altura)
    usando a caixa delimitadora mínima que envolve os pontos.
    """
    # Calcular a caixa delimitadora mínima (bounding box)
    points = np.array(points)
    x_min = np.min(points[:, 0])
    y_min = np.min(points[:, 1])
    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])

    # Calcular as coordenadas normalizadas para o YOLO
    x_center = (x_min + x_max) / 2 / width
    y_center = (y_min + y_max) / 2 / height
    w = (x_max - x_min) / width
    h = (y_max - y_min) / height

    # A classe pode ser definida de forma arbitrária (por exemplo, "0" para esse tipo de objeto)
    return [0, x_center, y_center, w, h]

# Função para processar cada arquivo de rótulo e corrigir o formato
def process_labels(label_path, image_shape):
    """
    Lê os rótulos de um arquivo e os converte para o formato YOLO.
    """
    corrected_labels = []
    
    # Tamanho da imagem para normalização
    height, width = image_shape[:2]

    try:
        with open(label_path, 'r') as file:
            lines = file.readlines()
        
        for line in lines:
            parts = line.strip().split()
            
            # Se a linha contiver mais de 1 coordenada, trata como um polígono
            if len(parts) > 5:
                points = [(float(parts[i]), float(parts[i+1])) for i in range(1, len(parts), 2)]
                yolo_label = convert_to_yolo_format_from_contour(points, width, height)
                corrected_labels.append(yolo_label)
            else:
                # Caso contrário, mantém o formato YOLO simples (classe, x_centro, y_centro, largura, altura)
                corrected_labels.append(list(map(float, parts)))
    except Exception as e:
        print(f"Erro ao processar o arquivo de rótulos {label_path}: {e}")
        return []

    # Salvar os rótulos corrigidos de volta no arquivo
    with open(label_path, 'w') as file:
        for label in corrected_labels:
            file.write(" ".join(map(str, label)) + '\n')

    return corrected_labels

# Caminhos para os dados
data_dir = 'D:/Kaue/Faculdade/TCC/TerceiraVersao/datasetRoadMark/train/images'  # Caminho para as imagens de treino
labels_dir = 'D:/Kaue/Faculdade/TCC/TerceiraVersao/datasetRoadMark/train/labels'  # Caminho para a pasta de rótulos

# Verificar se a pasta de rótulos existe e criar se necessário
if not os.path.exists(labels_dir):
    os.makedirs(labels_dir)
    print(f"Pasta de rótulos criada em: {labels_dir}")

train_images = []
train_labels = []

# Função para carregar as imagens e rótulos
def load_data(data_dir, labels_dir):
    images = []
    labels = []
    
    image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png'))]
    label_files = [f.replace('.jpg', '.txt').replace('.png', '.txt') for f in image_files]
    
    # Verificar se a quantidade de imagens e rótulos bate
    if len(image_files) != len(label_files):
        print(f"Erro: O número de imagens e rótulos não corresponde em {data_dir}.")
        print(f"Imagens encontradas: {len(image_files)}, Rótulos encontrados: {len(label_files)}")
        return [], []

    # Carregar as imagens e os rótulos
    for file_name in image_files:
        image_path = os.path.join(data_dir, file_name)
        label_path = os.path.join(labels_dir, file_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        # Carregar e redimensionar a imagem
        image = cv2.imread(image_path)
        if image is None:
            print(f"Erro: Não foi possível carregar a imagem {image_path}")
            continue
        image = cv2.resize(image, (416, 416))  # Redimensionar para 416x416
        image = image / 255.0  # Normalizar a imagem
        images.append(image)

        # Processar os rótulos para o formato YOLO
        labels_data = process_labels(label_path, image.shape)
        labels.append(labels_data)

    return images, labels

# Carregar os dados de treino
train_images, train_labels = load_data(data_dir, labels_dir)

# Verificar se as listas de rótulos e imagens não estão vazias
if not train_labels:
    print("Erro: Não foram encontrados rótulos de treino.")
    exit(1)

# Verificar se o número de imagens e rótulos bate
if len(train_images) != len(train_labels):
    print(f"Erro: O número de imagens de treino ({len(train_images)}) e rótulos ({len(train_labels)}) não é igual.")
    exit(1)  # Interrompe a execução do script se os dados estiverem inconsistentes

# Ajustar o tamanho das listas de rótulos para ter a mesma quantidade de objetos por imagem
def pad_labels(labels):
    max_labels = max(len(label) for label in labels)
    print(f"Máximo de rótulos por imagem: {max_labels}")

    # Preencher rótulos com [0, 0, 0, 0, 0] para imagens com menos rótulos
    padded_labels = []
    for label in labels:
        while len(label) < max_labels:
            label.append([0.0, 0.0, 0.0, 0.0, 0.0])  # Adiciona rótulos vazios
        padded_labels.append(label)
    
    return padded_labels

# Padronizar os rótulos de treino
train_labels = pad_labels(train_labels)

# Salvar os dados de treino
np.save('train_images.npy', np.array(train_images))
np.save('train_labels.npy', np.array(train_labels))

yaml_path = 'D:/Kaue/Faculdade/TCC/TerceiraVersao/datasetRoadMark/data.yaml'

print(f"Arquivo YAML salvo em {yaml_path}")

# Carregar o modelo pré-treinado YOLOv5
from ultralytics import YOLO
model = YOLO("yolov5s.pt")  # Usando YOLOv5s como modelo base

# Treinar o modelo
model.train(data=yaml_path, epochs=10, imgsz=416, batch=16)

# Salvar o modelo treinado
model.save('models/segundo_yolov5.pt')

print("Treinamento finalizado e modelo salvo.")
