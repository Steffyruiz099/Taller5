import numpy as np
import cv2
import glob
import os
import json

# criterios de terminacion
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# preparar puntos de objeto
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Matrices para almacenar puntos de objeto y puntos de imagen de todas las imágenes.
objpoints = []  # 3d punto en el espacio del mundo real
imgpoints = []  # 2d puntos en el plano de la imagen.

# Se llama la imagen, mediante la dirección de la imagen y el nombre de la imagen.
path = 'C:/Users/Steffany/Documents/Javeriana/Semestre 9/Procesamiento de Imagenes/imagenes/calibracion/celu'
path_file = os.path.join(path, '*.jpg')

# Se encuentran todos los nombres de ruta
images = glob.glob(path_file)

# Se crea una matriz con los nombres de las imágenes para guardarlas luego de marcar los puntos
nombres = ['1.jpg','2.jpg','3.jpg','4.jpg','5.jpg','6.jpg','7.jpg','8.jpg','9.jpg','10.jpg','11.jpg','12.jpg','13.jpg','14.jpg','15.jpg','16.jpg','17.jpg','18.jpg','19.jpg','20.jpg','21.jpg','22.jpg','23.jpg','24.jpg','25.jpg','26.jpg','27.jpg','28.jpg','29.jpg']
i = 0 # se inicializa la varible i en 0

for fname in images:
    img = cv2.imread(fname)                                             # se leen las imagenes de la ruta ingresada
    img = cv2.resize(img,(1280,720),interpolation = cv2.INTER_AREA)     # se cambia el tamaño de la imagen
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                        # se cambia la imagen a grises

    # Encuentra las esquinas del tablero de ajedrez
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

    # Si lo encuentra, agregue puntos de objeto, puntos de imagen (después de refinarlos)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Dibujar y mostrar las esquinas
        img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
        cv2.imwrite(nombres[i], img)
        cv2.imshow('img', img)
        cv2.waitKey(250)
        i = i + 1


cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(mtx)
print(dist)

# error de reproyección
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print("total error: {}".format(mean_error / len(objpoints)))

# distorsión
path_file = os.path.join(path, 'c2.jpg')
imgr = cv2.imread(path_file)
img = cv2.resize(imgr,(1280,720),interpolation = cv2.INTER_AREA)
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

if True:
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
else:
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# recortar la imagen
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv2.imwrite('distorted.jpg', img)           # se guarda la imagen de distorsión
cv2.imwrite('calibresult.jpg', dst)         # se guarda la imagen luego de ser calibrada
cv2.imshow('distorted', img)                # se muestra la imaden de distrosión
cv2.imshow('calibresult', dst)              # se muestra la imagen luego de ser calibrada
cv2.waitKey(0)

file_name = 'calibration.json'              # nombre de la imagen
json_file = os.path.join(path, file_name)

# se guardan los datos en el archivo json
data = {
    'K': mtx.tolist(),
    'distortion': dist.tolist()
}

with open(json_file, 'w') as fp:
    json.dump(data, fp, sort_keys=True, indent=1, ensure_ascii=False)

with open(json_file) as fp:
    json_data = json.load(fp)
print(json_data)
