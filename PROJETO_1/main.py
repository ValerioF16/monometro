from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas as rotas

@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        # Receber os parâmetros
        pa = float(request.form['pa'])
        pb = float(request.form['pb'])
        t = float(request.form['t'])

        # Receber a imagem
        if 'image' not in request.files:
            return jsonify({'error': 'Nenhuma imagem enviada'}), 400
        
        image_file = request.files['image']
        image_path = os.path.join(os.path.dirname(__file__), image_file.filename)
        image_file.save(image_path)

        # Carregar a imagem com OpenCV
        image = cv2.imread(image_path)
        if image is None:
            return jsonify({'error': 'Imagem não encontrada ou caminho incorreto.'}), 400

        # Processar a imagem
        height_left, height_right, height_diff_pixels = process_image(image)

        # Calcular a diferença de altura
        scale_factor = 0.1 / 100  
        height_left_m = height_left * scale_factor
        height_right_m = height_right * scale_factor
        height_diff_m = height_diff_pixels * scale_factor
        
        # Calcular a pressão
        rho = 1000  
        g = 9.81    
        pressure = rho * g * height_diff_m

        # Remover a imagem temporária
        os.remove(image_path)

        return jsonify({
            'pressure': pressure,
            'height_diff_m': height_diff_m,
            'Altura_PA': height_left_m,
            'Altura_PB': height_right_m
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
    except Exception as e:
        return jsonify({'error': 'Erro interno no servidor: ' + str(e)}), 500


def process_image(image):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    # Intervalo para a cor vermelha 
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Máscaras de vermelho
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    blurred = cv2.GaussianBlur(mask, (5, 5), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh_clean = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError('Nenhum contorno encontrado.')

    main_contour = max(contours, key=cv2.contourArea)

    y_bottom = max([point[0][1] for point in main_contour])

    x, y, w, h = cv2.boundingRect(main_contour)

    mid_x = x + w // 2
    roi_left = thresh_clean[:, :mid_x]
    roi_right = thresh_clean[:, mid_x:]

    contours_left, _ = cv2.findContours(roi_left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_right, _ = cv2.findContours(roi_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 1000
    contours_left = [c for c in contours_left if cv2.contourArea(c) > min_area]
    contours_right = [c for c in contours_right if cv2.contourArea(c) > min_area]

    if len(contours_left) == 0 or len(contours_right) == 0:
        raise ValueError('Não foram encontrados contornos suficientes')

    height_left = y_bottom - min([point[0][1] for point in contours_left[0]])
    height_right = y_bottom - min([point[0][1] for point in contours_right[0]])

    height_diff_pixels = abs(height_left - height_right)

    return height_left, height_right, height_diff_pixels


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
