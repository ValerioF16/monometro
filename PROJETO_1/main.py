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
        image_path = os.path.join('uploads', image_file.filename)
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

        # Remover a imagem temporária (opcional)
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

# Define o intervalo para a cor vermelha 
    lower_red1 = np.array([0, 120, 70])    # Intervalo inferior para vermelho
    upper_red1 = np.array([10, 255, 255])  # Intervalo superior
    lower_red2 = np.array([170, 120, 70])  # Segundo intervalo (vermelho escuro)
    upper_red2 = np.array([180, 255, 255])

# Cria máscaras para os intervalos de vermelho
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)

# Operação morfológica para melhorar a segmentação
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh_clean = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

    # Encontra o contorno principal na imagem completa
    contours, _ = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("Nenhum contorno encontrado.")
        exit()

    # Pega o maior contorno (fluido completo)
    main_contour = max(contours, key=cv2.contourArea)

    # Encontra o ponto mais baixo do fluido (centro na região inferior)
    y_bottom = max([point[0][1] for point in main_contour])
    x_bottom = [point[0][0] for point in main_contour if point[0][1] == y_bottom][0]
    center_bottom = (x_bottom, y_bottom)

    # Encontra o retângulo delimitador do contorno principal
    x, y, w, h = cv2.boundingRect(main_contour)

    # Divide a imagem em duas ROIs: esquerda e direita
    mid_x = x + w // 2
    roi_left = thresh_clean[:, :mid_x]
    roi_right = thresh_clean[:, mid_x:]

    # Encontra contornos em cada ROI
    contours_left, _ = cv2.findContours(roi_left, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_right, _ = cv2.findContours(roi_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtra contornos pequenos
    min_area = 1000  # Ajuste conforme necessário
    contours_left = [c for c in contours_left if cv2.contourArea(c) > min_area]
    contours_right = [c for c in contours_right if cv2.contourArea(c) > min_area]

    if len(contours_left) == 0 or len(contours_right) == 0:
        print("Não foram encontrados contornos suficientes nas ROIs.")
        raise ValueError("Não foram encontrados contornos suficientes")

    # Pega o maior contorno em cada ROI
    contour_left = max(contours_left, key=cv2.contourArea)
    contour_right = max(contours_right, key=cv2.contourArea)

    # Ajusta as coordenadas do contorno direito para a imagem original
    contour_right_adjusted = contour_right.copy()
    for point in contour_right_adjusted:
        point[0][0] += mid_x

    # Encontra as alturas mínima e máxima de cada contorno
    # Braço esquerdo
    y_left_min = y_bottom  # Mínimo é o ponto mais baixo do fluido
    y_left_max = min([point[0][1] for point in contour_left])  # Máximo é o topo
    height_left = y_left_min - y_left_max  # Altura em pixels

    # Braço direito
    y_right_min = y_bottom  # Mínimo é o ponto mais baixo do fluido
    y_right_max = min([point[0][1] for point in contour_right_adjusted])  # Máximo é o topo
    height_right = y_right_min - y_right_max  # Altura em pixels
    # Diferença de altura entre os topos
    height_diff_pixels = abs(y_left_max - y_right_max)

    return height_left, height_right, height_diff_pixels  

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True, host='0.0.0.0', port=5000)