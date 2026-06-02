import cv2

# Carga el archivo de video
video_path = r"D:\Proyectos\Sistema ILSC\Entrenamiento_modelos_colsign\dataset_videos\A veces\A veces_0e5ffbcb-e68c-4d53-904c-bd0e3685bbd2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error al abrir el video.")
else:
    # Obtener los FPS de los metadatos
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"El video corre a: {fps} FPS")

# No olvides liberar el objeto
cap.release()