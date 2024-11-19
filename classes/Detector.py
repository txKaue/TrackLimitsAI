import cv2
import numpy as np
from ultralytics import YOLO
import time

class TrackLimitDetector:
    def __init__(self, model_path, output_video_path, log_file_path):
        # Carregar o modelo treinado YOLO
        self.model = YOLO(model_path)
        self.output_video_path = output_video_path
        self.log_file_path = log_file_path

    def processar_video(self, caminho_video):
        # Abre o vídeo a partir do caminho fornecido
        cap = cv2.VideoCapture(caminho_video)

        if not cap.isOpened():
            print("Erro ao abrir o vídeo.")
            return

        # Obter as propriedades do vídeo
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Configurar o escritor de vídeo para salvar o resultado
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codificação para o arquivo .mp4
        out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))

        # Criar ou abrir o arquivo de log para armazenar os momentos das detecções
        with open(self.log_file_path, 'w') as log_file:
            log_file.write("Momentos das detecções (em segundos):\n")

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                try:
                    # Redimensionar o frame para 416x416 para o modelo YOLO
                    resized_frame = cv2.resize(frame, (416, 416))

                    # Realizar a predição com o modelo YOLO
                    results = self.model(resized_frame)  # Faz a predição no frame
                    
                    # Obter as caixas delimitadoras e as confidências
                    pred_boxes = results[0].boxes.xywh.cpu().numpy()  # Coordenadas das caixas
                    pred_scores = results[0].boxes.conf.cpu().numpy()  # Confiabilidade das predições
                    pred_classes = results[0].boxes.cls.cpu().numpy()  # Classes das predições

                    # Verificar se foram feitas detecções
                    detected = False
                    for box, score, cls in zip(pred_boxes, pred_scores, pred_classes):
                        if score > 0.5:  # Considera predições com confiança maior que 0.5
                            x_center, y_center, width, height = box
                            x1, y1 = int((x_center - width / 2) * frame.shape[1]), int((y_center - height / 2) * frame.shape[0])
                            x2, y2 = int((x_center + width / 2) * frame.shape[1]), int((y_center + height / 2) * frame.shape[0])

                            # Desenha a caixa delimitadora no vídeo
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Caixa verde
                            cv2.putText(frame, f'Class: {int(cls)} Conf: {score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                            # Marcar que houve uma detecção
                            detected = True

                    # Se houver detecção, registrar o tempo (em segundos) no arquivo de log
                    if detected:
                        timestamp = frame_count / fps  # Tempo em segundos
                        log_file.write(f"{timestamp:.2f} segundos\n")

                    # Escrever o frame processado no vídeo de saída
                    out.write(frame)

                except Exception as e:
                    print(f"Erro ao processar o frame: {e}")

            cap.release()
            out.release()

        print(f"Processamento finalizado. Vídeo salvo em: {self.output_video_path}")
        print(f"Momentos das detecções salvos em: {self.log_file_path}")

# Caminhos de entrada e saída
video_path = 'D:/Kaue/Faculdade/TCC/TerceiraVersao/images/AustriaMaxCorte.mp4'  # Alterar para o seu vídeo
model_path = 'models/segundo_yolov5.pt'  # Caminho para o modelo treinado
output_video_path = 'D:/Kaue/Faculdade/TCC/TerceiraVersao/images/video_processado.mp4'  # Caminho para o vídeo de saída
log_file_path = 'D:/Kaue/Faculdade/TCC/TerceiraVersao/images/detecoes.txt'  # Caminho para o arquivo de log

# Inicializar o detector
detector = TrackLimitDetector(model_path, output_video_path, log_file_path)

# Processar o vídeo
detector.processar_video(video_path)
