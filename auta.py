import cv2
import numpy as np

# Wczytaj konfigurację i wytrenowane wagi YOLOv4
net = cv2.dnn.readNet("yolov4.cfg", "yolov4.weights")

# Odczytaj plik z nazwami klas
with open("coco_bez_pl.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getUnconnectedOutLayersNames()

# Wczytaj plik wideo
cap = cv2.VideoCapture("output.mp4")

 # Ustal rozmiar klatki
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = 20

# Inicjalizacja VideoWriter
out1 = cv2.VideoWriter('rozpoznane.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while True:
    # Pobieranie obrazu z pliku
    ret, frame = cap.read()
    if not ret:
        break
    height, width, _ = frame.shape

   # Przetwarzanie obrazu przez sieć YOLOv4
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Przetwarzanie wyników detekcji
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == 'Samochod':

               
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Filtrowanie detekcji z użyciem tresholdu
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

     # Wyświetlanie wyników
    font = cv2.FONT_HERSHEY_PLAIN
    object_count = {}

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, label, (x, y + 30), font, 1, (0, 0, 255), 2)

            # Liczenie ilości obiektów danej klasy
            if label in object_count:
                object_count[label] += 1
            else:
                object_count[label] = 1

    
    # Wyświetl obraz na ekranie
    cv2.imshow('Detekcja aut', frame)

    # Zapisz klatkę do pliku
    out1.write(frame)        

    # Przerwij pętlę po naciśnięciu klawisza 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zwolnij zasoby
out1.release()
cap.release()

cv2.destroyAllWindows()

# Wyświetlanie liczby obiektów klasy Samochod
for label, count in object_count.items():
    print(f"Ilość obiektów klasy {label}: {count}")
