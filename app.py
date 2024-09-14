import torch
from flask import Flask, request, jsonify
from PIL import Image
import io
from ultralytics import YOLO

# YOLO 모델 가져오기
model = YOLO('/Users/iseul-a/Downloads/helmet_result/best-2.pt')
app = Flask(__name__)
@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # 업로드된 이미지 가져오기
    image_file = request.files['image']
    image_bytes = image_file.read()
    img = Image.open(io.BytesIO(image_bytes))

    # YOLO 모델로 이미지 처리
    results = model(img)

    # 결과를 딕셔너리로 변환 (YOLO 모델의 results를 처리)
    processed_data = results.pandas().xyxy[0].to_dict(orient="records")

    return jsonify({"detections": processed_data})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
