from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from channels.generic.websocket import AsyncWebsocketConsumer
import cv2
import os
import face_recognition
import asyncio

prototxt = "/home/edmartinez/Documents/UTPL/Septimo Ciclo/Inteligencia Artificial/DeteccionFacialGUI/FaceRecognition/deteccion/static/model/deploy.prototxt" # Arquitectura
model = "/home/edmartinez/Documents/UTPL/Septimo Ciclo/Inteligencia Artificial/DeteccionFacialGUI/FaceRecognition/deteccion/static/model/res10_300x300_ssd_iter_140000.caffemodel" # Pesos
haarcascade = "/home/edmartinez/Documents/UTPL/Septimo Ciclo/Inteligencia Artificial/DeteccionFacialGUI/FaceRecognition/deteccion/static/model/haarcascade_frontalface_default.xml"
net = cv2.dnn.readNetFromCaffe(prototxt, model)# Cargamos el modelo de la red

def index(request):
    return render(request, 'principal.html')


def principal_view(request):
    return render(request, 'principal.html')

def train_model(request):
    
    if request.method == 'POST' and request.FILES['train_image']:
        # Obtiene la imagen desde el formulario
        uploaded_image = request.FILES['train_image']
        nombre = request.POST['nombre']
        image_path = os.path.join(settings.MEDIA_ROOT, 'Uploads', uploaded_image.name)

        if not os.path.exists(os.path.dirname(image_path)):
            os.makedirs(os.path.dirname(image_path))

        # Guarda la imagen en la carpeta "uploads"
        with open(image_path, 'wb') as destination:
            for chunk in uploaded_image.chunks():
                destination.write(chunk)

        image = cv2.imread(image_path)
        height, width, _ = image.shape
        image_resized = cv2.resize(image, (300, 300))

        # Create a blob
        blob = cv2.dnn.blobFromImage(image_resized, 1.0, (300, 300), (104, 117, 123))
        print("blob.shape: ", blob.shape)
        blob_to_show = cv2.merge([blob[0][0], blob[0][1], blob[0][2]])

        net.setInput(blob)
        detections = net.forward()
        print("detections.shape:", detections.shape)

        for idx, detection in enumerate(detections[0][0]):
            if detection[2] > 0.5:
                box = detection[3:7] * [width, height, width, height]
                x_start, y_start, x_end, y_end = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                face = image[y_start:y_end, x_start:x_end]
                face = cv2.resize(face, (150, 150))

                # Save the detected face as an image
                face_filename = os.path.join(settings.MEDIA_ROOT, 'RostrosDetectados', (nombre+f"{idx}.jpg"))

                if not os.path.exists(os.path.dirname(face_filename)):
                    os.makedirs(os.path.dirname(face_filename))

                cv2.imwrite(face_filename, face)

                cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
                cv2.putText(image, "Conf: {:.2f}".format(detection[2] * 100), (x_start, y_start - 5), 1, 1.2, (0, 255, 255), 2)

        cv2.destroyAllWindows()
    
    return render(request, 'principal.html')

def activar_camera(request):
    
    faces_encodings, faces_names = get_faces_encodings()

    cap = cv2.VideoCapture(0, cv2.CAP_ANY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        orig = frame.copy()
        faces = face_cascade.detectMultiScale(frame, 1.1, 5)

        for (x, y, w, h) in faces:
            face = orig[y:y + h, x:x + w]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            actual_face_encoding = face_recognition.face_encodings(face, known_face_locations=[(0, w, h, 0)])[0]
            result = face_recognition.compare_faces(faces_encodings, actual_face_encoding)

            if True in result:
                index = result.index(True)
                name = faces_names[index]
                color = (125, 220, 0)
            else:
                name = "Desconocido"
                color = (50, 50, 255)

            cv2.rectangle(frame, (x, y + h), (x + w, y + h + 30), color, -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y + h + 25), 2, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()

    return render(request, 'principal.html')

def detect_people(request):
    if request.method == 'POST' and request.FILES['detect_image']:
        # Obtiene la imagen desde el formulario
        uploaded_image = request.FILES['detect_image']
        image_path = os.path.join(settings.MEDIA_ROOT, 'Detecciones', uploaded_image.name)

        # Guarda la imagen en la carpeta "uploads"
        with open(image_path, 'wb') as destination:
            for chunk in uploaded_image.chunks():
                destination.write(chunk)

        faces_encodings, faces_names = get_faces_encodings()

        frame = cv2.imread(image_path)
        orig = frame.copy()
        faces = face_recognition.face_locations(frame)

        for (top, right, bottom, left) in faces:
            face = orig[top:bottom, left:right]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            actual_face_encoding = face_recognition.face_encodings(face)

            if actual_face_encoding:
                # Si se detecta una cara, realiza la comparación
                result = face_recognition.compare_faces(faces_encodings, actual_face_encoding[0])

                if True in result:
                    index = result.index(True)
                    name = faces_names[index]
                else:
                    name = "Desconocido"
            else:
                # Si no se detecta ninguna cara, asigna el nombre como "Desconocido"
                name = "Desconocido"

            # Muestra el nombre en el cuadro del rostro
            if (name == "Desconocido"):
                color = (0, 0, 255)
            else:
                color = (125, 220, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, bottom + 25), 2, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #return HttpResponse('Personas detectadas en la imagen.')

    return render(request, 'principal.html')

def get_faces_encodings():
    image_faces_path = "/home/edmartinez/Documents/UTPL/Septimo Ciclo/Inteligencia Artificial/DeteccionFacialV2/RostrosDetectados"
    faces_encodings = []
    faces_names = []

    for file_name in os.listdir(image_faces_path):
        image = cv2.imread(os.path.join(image_faces_path, file_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        f_coding = face_recognition.face_encodings(image, known_face_locations=[(0, 150, 150, 0)])[0]
        faces_encodings.append(f_coding)
        faces_names.append(file_name.split(".")[0])

    return faces_encodings, faces_names