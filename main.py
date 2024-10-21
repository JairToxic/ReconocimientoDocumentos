import cv2
import pytesseract
import torch

# Configura la ruta a Tesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Cargar el modelo YOLOv5 preentrenado
print("Cargando modelo YOLOv5...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Leer la imagen
img_path = './OCR.png'  # Cambia esta ruta a tu imagen
img = cv2.imread(img_path)

# Verificar si la imagen se ha cargado correctamente
if img is None:
    print("Error al cargar la imagen.")
    exit()

# Preprocesar la imagen
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blurred, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)

# Aplicar OCR a la imagen completa
custom_config = r'--oem 3 --psm 6'
full_text = pytesseract.image_to_string(thresh, config=custom_config)
print(f'Texto completo detectado: {full_text.strip()}')

# Realizar la detección
print("Iniciando detección...")
results = model(img)
print("Detección completada.")
print(f'Resultados de detección: {results.xyxy[0]}')

# Extraer las cajas delimitadoras y aplicar OCR
boxes = results.xyxy[0]  # Coordenadas de las cajas

for box in boxes:
    x1, y1, x2, y2, conf, cls = box
    if conf > 0.5:
        # Extraer la región de interés (ROI)
        roi = img[int(y1):int(y2), int(x1):int(x2)]
        
        # Aplicar OCR a la ROI
        roi_text = pytesseract.image_to_string(roi, config=custom_config)
        print(f'Detectado texto en ROI: {roi_text.strip()}')

        # Dibujar las cajas en la imagen original
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, roi_text.strip(), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Mostrar la imagen con las detecciones
cv2.imshow("Resultados de Detección", img)
cv2.waitKey(0)  # Esperar hasta que se presione una tecla
cv2.destroyAllWindows()  # Cierra todas las ventanas al final
