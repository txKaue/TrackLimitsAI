import os
import cv2
import numpy as np
from ultralytics import YOLO

# Caminhos para os dados de validação
val_images_dir = 'D:/Kaue/Faculdade/TCC/TerceiraVersao/datasetRoadMark/val/images'  # Imagens de validação
val_labels_dir = 'D:/Kaue/Faculdade/TCC/TerceiraVersao/datasetRoadMark/val/labels'  # Rótulos de validação

# Caminho do modelo treinado
model_path = 'models/segundo_yolov5.pt'

# Função para carregar e processar dados de validação
def load_validation_data(images_dir, labels_dir):
    images = []
    labels = []

    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
    label_files = [f.replace('.jpg', '.txt').replace('.png', '.txt') for f in image_files]

    if len(image_files) != len(label_files):
        print(f"Erro: O número de imagens e rótulos não corresponde em {images_dir}.")
        print(f"Imagens encontradas: {len(image_files)}, Rótulos encontrados: {len(label_files)}")
        return [], []

    for file_name in image_files:
        image_path = os.path.join(images_dir, file_name)
        label_path = os.path.join(labels_dir, file_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        # Carregar e redimensionar a imagem
        image = cv2.imread(image_path)
        if image is None:
            print(f"Erro: Não foi possível carregar a imagem {image_path}")
            continue
        image = cv2.resize(image, (416, 416))  # Redimensionar para 416x416
        image = image / 255.0  # Normalizar a imagem
        images.append(image)

        # Carregar os rótulos
        try:
            with open(label_path, 'r') as file:
                label_data = []
                lines = file.readlines()
                for line in lines:
                    label_data.append(list(map(float, line.strip().split())))
                labels.append(label_data)
        except Exception as e:
            print(f"Erro ao processar o rótulo {label_path}: {e}")
            labels.append([])  # Adiciona rótulo vazio em caso de erro

    return np.array(images), labels

# Carregar os dados de validação
val_images, val_labels = load_validation_data(val_images_dir, val_labels_dir)

# Verificar se os dados foram carregados corretamente
if len(val_images) == 0 or len(val_labels) == 0:
    print("Erro: Dados de validação não carregados corretamente.")
    exit(1)

# Carregar o modelo treinado
model = YOLO(model_path)

# Função para calcular métricas
def evaluate_model(model, images, labels):
    """
    Avalia o modelo usando as imagens e os rótulos de validação.
    """
    total_images = len(images)
    total_predictions = 0
    total_ground_truths = 0
    true_positives = 0

    for i, image in enumerate(images):
        # Fazer a previsão para a imagem
        results = model.predict(image, imgsz=416, conf=0.25, verbose=False)
        predictions = results[0].boxes.xywh  # Pega as caixas preditas no formato [x_centro, y_centro, largura, altura]
        confidences = results[0].boxes.conf  # Confianças das predições

        # Calcular métricas com base nas predições e nos rótulos
        ground_truths = labels[i]
        total_predictions += len(predictions)
        total_ground_truths += len(ground_truths)

        # Calcular verdadeiros positivos com IoU > 0.5
        for pred_box in predictions:
            for gt_box in ground_truths:
                iou = calculate_iou(pred_box, gt_box)
                if iou > 0.5:
                    true_positives += 1
                    break  # Evita contar múltiplas predições para o mesmo ground truth

    precision = true_positives / total_predictions if total_predictions > 0 else 0
    recall = true_positives / total_ground_truths if total_ground_truths > 0 else 0
    return precision, recall

# Função para calcular o IoU (Intersection over Union)
def calculate_iou(box1, box2):
    """
    Calcula o IoU entre duas caixas delimitadoras.
    As caixas devem estar no formato [x_centro, y_centro, largura, altura].
    """
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

# Avaliar o modelo
precision, recall = evaluate_model(model, val_images, val_labels)

print(f"Precisão: {precision:.4f}")
print(f"Abrangência (Recall): {recall:.4f}")
