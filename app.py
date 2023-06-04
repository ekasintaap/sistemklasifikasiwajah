from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__, static_url_path='/static')

app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.JPG', '.png', '.PNG', '.JPEG']
app.config['UPLOAD_PATH'] = './static/images/uploads/'

model = None

NUM_CLASSES = 5
kelas = ["Acne", "Flek Hitam", "Healthy", "Rosacea", "Panu"]




# Load model saat aplikasi Flask dijalankan
@app.before_first_request
def load_model_():
    global model
    model = load_model("modelkulitwajah94.h5")

# Routing untuk halaman utama atau home
@app.route("/")
def beranda():
    return render_template('index.html')

# Routing untuk API deteksi
@app.route("/api/deteksi", methods=['POST'])
def apiDeteksi():
    # Set nilai default untuk hasil prediksi dan gambar yang diprediksi
    hasil_prediksi = '(none)'
    gambar_prediksi = '(none)'

    # Get File Gambar yang telah diupload oleh pengguna
    uploaded_file = request.files['file']
    filename = secure_filename(uploaded_file.filename)

    # Periksa apakah ada file yang dipilih untuk diupload
    if filename != '':
        # Set / mendapatkan extension dan path dari file yang diupload
        file_ext = os.path.splitext(filename)[1]
        gambar_prediksi = '/static/images/uploads/' + filename

        # Periksa apakah extension file yang diupload sesuai (jpg)
        if file_ext in app.config['UPLOAD_EXTENSIONS']:
            # Simpan gambar
            uploaded_file.save(os.path.join(app.config['UPLOAD_PATH'], filename))

            # Memuat gambar
            test_image = Image.open('.' + gambar_prediksi)

            # Mengubah ukuran gambar
            test_image_resized = test_image.resize((224, 224))

            # Konversi gambar ke array
            image_array = np.array(test_image_resized)
            test_image_x = (image_array / 255)
            test_image_x = np.expand_dims(test_image_x, axis=0)

            # Prediksi gambar
            y_pred_test_single = model.predict(test_image_x)
            y_pred_test_classes_single = np.argmax(y_pred_test_single, axis=1)

            hasil_prediksi = kelas[y_pred_test_classes_single[0]]

        # Return hasil prediksi dengan format JSON
        return jsonify({
            "prediksi": hasil_prediksi,
            "gambar_prediksi": gambar_prediksi
        })
         

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    
    # run_with_ngrok(app)
    app.run(host="0.0.0.0", port=5001, debug=True)
