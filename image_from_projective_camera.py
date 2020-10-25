import cv2
from camera_model import *
import json

if __name__ == '__main__':
    file_name = 'C:/Users/karen/Documents/Ingenieriaelectronica/NovenoSemestre/Procesamientodeimagenes/CamaraComputador/calibration.json'
    #Se lee el archivo.js
    with open(file_name) as fp:
        json_data = json.load(fp)

    # intrinsics parameters
    K = np.array(json_data['K'])
    width = int(K[0][2] * 2)
    height = int(K[1][2] * 2)
    print(width, height)

    #extrinsics parameters
    R = json_data['rotation']
    R = set_rotation(R[0][0],R[0][1],R[0][2])
    t = json_data['traslation']
    t = np.array(t[0])

    
    # create camera
    camera = projective_camera(K, width, height, R, t)

    # Puntos del cubo en 3D
    square_3D = np.array([[0.5, 0.5, 0], [0.5, -0.5, 0], [-0.5, -0.5, 0], [-0.5, 0.5, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5]])
    # Pixeles x, y de las esquinas del cubo
    square_2D = projective_camera_project(square_3D, camera)
    image_projective = 255 * np.ones(shape=[camera.height, camera.width, 3], dtype=np.uint8)
    #Se dibujan las 12 líneas de cubo
    cv2.line(image_projective, (square_2D[0][0], square_2D[0][1]), (square_2D[1][0], square_2D[1][1]), (200, 1, 255), 3)
    cv2.line(image_projective, (square_2D[1][0], square_2D[1][1]), (square_2D[2][0], square_2D[2][1]), (200, 1, 255), 3)
    cv2.line(image_projective, (square_2D[2][0], square_2D[2][1]), (square_2D[3][0], square_2D[3][1]), (200, 1, 255), 3)
    cv2.line(image_projective, (square_2D[3][0], square_2D[3][1]), (square_2D[0][0], square_2D[0][1]), (200, 1, 255), 3)
    cv2.line(image_projective, (square_2D[4][0], square_2D[4][1]), (square_2D[5][0], square_2D[5][1]), (200, 1, 255), 3)
    cv2.line(image_projective, (square_2D[5][0], square_2D[5][1]), (square_2D[6][0], square_2D[6][1]), (200, 1, 255), 3)
    cv2.line(image_projective, (square_2D[6][0], square_2D[6][1]), (square_2D[7][0], square_2D[7][1]), (200, 1, 255), 3)
    cv2.line(image_projective, (square_2D[7][0], square_2D[7][1]), (square_2D[4][0], square_2D[4][1]), (200, 1, 255), 3)
    cv2.line(image_projective, (square_2D[0][0], square_2D[0][1]), (square_2D[4][0], square_2D[4][1]), (200, 1, 255), 3)
    cv2.line(image_projective, (square_2D[1][0], square_2D[1][1]), (square_2D[5][0], square_2D[5][1]), (200, 1, 255), 3)
    cv2.line(image_projective, (square_2D[2][0], square_2D[2][1]), (square_2D[6][0], square_2D[6][1]), (200, 1, 255), 3)
    cv2.line(image_projective, (square_2D[3][0], square_2D[3][1]), (square_2D[7][0], square_2D[7][1]), (200, 1, 255), 3)
    cv2.imwrite('respuesta2.jpg',image_projective) #Se guarda
    cv2.imshow("Image", image_projective) #Se muestra
    cv2.waitKey(0)

