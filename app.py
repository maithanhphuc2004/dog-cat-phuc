from flask import Flask, request, render_template
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Tải mô hình
MODEL_PATH = "model/model.h5"
model = load_model(MODEL_PATH)

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128)) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array /= 255.0  
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    img_path = None

    if request.method == "POST":
        try:
            file = request.files.get('file')
            if file:
                upload_folder = "static/uploads"
                if not os.path.exists(upload_folder):
                    os.makedirs(upload_folder)
                
                img_path = os.path.join(upload_folder, file.filename)
                file.save(img_path)

                # Dự đoán
                processed_img = prepare_image(img_path)
                result = model.predict(processed_img)
                
                # result
                raw_score = result[0][0]
                
                if raw_score > 0.5:
                    prediction = "Chó (Dog)"
                    confidence = round(raw_score * 100, 2)
                else:
                    prediction = "Mèo (Cat)"
                    # Vì là Mèo nên xác suất sẽ là (1 - raw_score)
                    confidence = round((1 - raw_score) * 100, 2)
                
                return render_template("index.html", 
                                       prediction=prediction, 
                                       confidence=confidence, 
                                       img_path=img_path)
        except Exception as e:
            prediction = f"Lỗi: {str(e)}"
            
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)