import torch
from flask import Flask, request, jsonify , render_template_string
from PIL import Image
import io
from ultralytics import YOLO

# YOLO 모델 가져오기
model = YOLO('/Users/iseul-a/Downloads/아카이브/best-2.pt')
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
    results = model(img, conf=0.25)  # 임계값 조정

    # 라벨링 정보만 추출
    labels = []
    for result in results:
        for i in range(len(result.boxes.cls)):  # 감지된 객체 수만큼 반복
            class_id = int(result.boxes.cls[i])  # 클래스 ID 가져오기

            # 헬멧 착용 여부에 따라 라벨링
            if class_id == 0:
                # 직접 HTML을 반환하여 클라이언트 측에서 페이지 이동 처리
                return render_template_string('''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Redirecting...</title>
                    <script>
                        window.location.href = "http://localhost:8080/success";
                    </script>
                </head>
                <body>
                </body>
                </html>
                ''')
            elif class_id == 1:
                label = "No Helmet"
            labels.append(label)

    # 감지된 라벨 리스트를 JSON으로 반환
    return jsonify({"labels": labels})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)