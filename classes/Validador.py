import os
import cv2
import numpy as np
from ultralytics import YOLO

# Caminhos para os dados de validação
val_images_dir = "D:/Kaue/Faculdade/TCC/TerceiraVersao/datasetRoadMark/valid/images"
val_labels_dir = "D:/Kaue/Faculdade/TCC/TerceiraVersao/datasetRoadMark/valid/labels"

# Caminho do modelo treinado
model_path = "models/segundo_yolov5.pt"

# Função para converter rótulos para o formato YOLO
def convert_labels_to_yolo(label_path, image_shape):
    """
    Converte rótulos de contornos para o formato YOLO.
    """
    corrected_labels = []
    height, width = image_shape[:2]

    try:
        with open(label_path, 'r') as file:
            lines = file.readlines()
        
        for line in lines:
            parts = line.strip().split()

            # Se a linha contiver mais de 5 elementos, é um polígono
            if len(parts) > 5:
                points = [(float(parts[i]), float(parts[i + 1])) for i in range(1, len(parts), 2)]
                x_min = min(p[0] for p in points)
                y_min = min(p[1] for p in points)
                x_max = max(p[0] for p in points)
                y_max = max(p[1] for p in points)

                x_center = (x_min + x_max) / 2 / width
                y_center = (y_min + y_max) / 2 / height
                w = (x_max - x_min) / width
                h = (y_max - y_min) / height
                corrected_labels.append([0, x_center, y_center, w, h])
            else:
                # Formato YOLO já válido
                corrected_labels.append(list(map(float, parts)))
    except Exception as e:
        print(f"Erro ao processar rótulo {label_path}: {e}")
        return []

    return corrected_labels

# Função para carregar e processar dados de validação
def load_validation_data(images_dir, labels_dir):
    images = []
    labels = []

    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
    label_files = [f.replace('.jpg', '.txt').replace('.png', '.txt') for f in image_files]

    for file_name in image_files:
        image_path = os.path.join(images_dir, file_name)
        label_path = os.path.join(labels_dir, file_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        # Carregar e redimensionar a imagem
        image = cv2.imread(image_path)
        if image is None:
            print(f"Erro: Não foi possível carregar a imagem {image_path}")
            continue
        image = cv2.resize(image, (416, 416))
        image = image / 255.0  # Normalizar a imagem
        images.append(image)

        # Processar os rótulos
        labels.append(convert_labels_to_yolo(label_path, image.shape))

    return np.array(images), labels

# Função para calcular o IoU (Intersection over Union)
def calculate_iou(box1, box2):
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

# Função para calcular métricas
def evaluate_model(model, images, labels):
    total_images = len(images)
    total_predictions = 0
    total_ground_truths = 0
    true_positives = 0

    for i, image in enumerate(images):
        results = model.predict(image, imgsz=416, conf=0.25, verbose=False)
        predictions = results[0].boxes.xywh.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()

        ground_truths = labels[i]
        total_predictions += len(predictions)
        total_ground_truths += len(ground_truths)

        # Calcular verdadeiros positivos com IoU > 0.5
        for pred_box in predictions:
            for gt_box in ground_truths:
                iou = calculate_iou(pred_box, gt_box[1:])
                if iou > 0.5:
                    true_positives += 1
                    break

    precision = true_positives / total_predictions if total_predictions > 0 else 0
    recall = true_positives / total_ground_truths if total_ground_truths > 0 else 0
    return precision, recall

# Carregar os dados de validação
val_images, val_labels = load_validation_data(val_images_dir, val_labels_dir)

# Carregar o modelo treinado
model = YOLO(model_path)

# Avaliar o modelo
precision, recall = evaluate_model(model, val_images, val_labels)

print(f"Precisão: {precision:.4f}")
print(f"Abrangência (Recall): {recall:.4f}")
