import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Caminhos para os dados e modelo
data_dir = 'D:/Kaue/Faculdade/TCC/TerceiraVersao/images/validation/'  # Caminho para as imagens de validação
labels_dir = 'D:/Kaue/Faculdade/TCC/TerceiraVersao/labels/validation/'  # Caminho para os rótulos de validação
model_path = 'models/cantil_yolov5.pt'  # Caminho para o modelo treinado

# Carregar o modelo treinado
model = YOLO(model_path)  # Use o arquivo .pt

# Função para carregar as imagens e rótulos de validação
def load_data(data_dir, labels_dir):
    images = []
    labels = []
    
    image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png'))]
    label_files = [f.replace('.jpg', '.txt').replace('.png', '.txt') for f in image_files]

    # Carregar as imagens e rótulos
    for file_name in image_files:
        image_path = os.path.join(data_dir, file_name)
        label_path = os.path.join(labels_dir, file_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        # Carregar e redimensionar a imagem
        image = cv2.imread(image_path)
        if image is None:
            print(f"Erro ao carregar a imagem {image_path}")
            continue
        image = cv2.resize(image, (416, 416))  # Redimensionar para 416x416
        images.append(image)

        # Carregar rótulos no formato YOLO
        try:
            with open(label_path, 'r') as file:
                label = [list(map(float, line.strip().split())) for line in file]
                labels.append(label)
        except Exception as e:
            print(f"Erro ao carregar rótulos de {label_path}: {e}")
            continue

    return images, labels

# Carregar os dados de validação
val_images, val_labels = load_data(data_dir, labels_dir)

# Verificar se os dados de validação foram carregados corretamente
if not val_labels:
    print("Erro: Não foram encontrados rótulos de validação.")
    exit(1)

if len(val_images) != len(val_labels):
    print(f"Erro: O número de imagens de validação ({len(val_images)}) e rótulos ({len(val_labels)}) não é igual.")
    exit(1)

# Função para validar o modelo
def validate_model(model, val_images, val_labels):
    predictions = []
    true_labels = []

    # Processar as imagens de validação
    for image, labels in zip(val_images, val_labels):
        # Realizar predições com o modelo YOLOv5
        results = model(image)  # Isso retorna uma lista de predições
        
        # Visualizar as predições
        print("Predictions:", results)
        
        # Filtrando as predições com uma confiança mínima de 0.5
        pred_boxes = results[0].boxes.xywh.cpu().numpy()
        pred_scores = results[0].boxes.conf.cpu().numpy()
        pred_classes = results[0].boxes.cls.cpu().numpy()

        print("Pred Boxes:", pred_boxes)
        print("Pred Scores:", pred_scores)
        print("Pred Classes:", pred_classes)

        # Verificar se as predições estão com confiança maior que 0.5
        valid_preds = [(cls, score) for cls, score in zip(pred_classes, pred_scores) if score > 0.5]
        print("Valid Predictions:", valid_preds)

        # Adicionar predições válidas à lista de predições
        predictions.append(valid_preds)

        # Adicionar os rótulos verdadeiros (classe) à lista
        true_labels.append([label[0] for label in labels])  # Considera a classe no formato YOLO

    return predictions, true_labels

# Função para calcular métricas de performance
def calculate_metrics(predictions, true_labels):
    all_true = []  # Lista para armazenar todos os rótulos verdadeiros
    all_pred = []  # Lista para armazenar todas as predições

    for pred, true in zip(predictions, true_labels):
        if len(pred) > 0:  # Verifica se há predições válidas
            valid_preds = [p[0] for p in pred]  # Usa a classe das predições válidas
            all_pred.extend(valid_preds)
            all_true.extend(true)
        else:
            print("No valid predictions for this sample.")

    if len(all_true) == 0 or len(all_pred) == 0:
        print("Não há predições ou rótulos válidos para calcular as métricas!")
        return 0, 0, 0  # Retorna valores padrão se não houver predições ou rótulos válidos
    
    # Calcular as métricas (ajustando conforme necessário, por exemplo, para multi-classe)
    precision = precision_score(all_true, all_pred, average='weighted', zero_division=1)
    recall = recall_score(all_true, all_pred, average='weighted', zero_division=1)
    f1 = f1_score(all_true, all_pred, average='weighted', zero_division=1)

    return precision, recall, f1

# Realizar a validação
predictions, true_labels = validate_model(model, val_images, val_labels)

# Calcular e exibir as métricas
precision, recall, f1 = calculate_metrics(predictions, true_labels)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
