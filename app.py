import os
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from datetime import timedelta
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------- CONFIG -------------------
app = Flask(__name__)
app.secret_key = "super_secret_key"
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
app.permanent_session_lifetime = timedelta(hours=6)
DATABASE = "doctors.db"

# ------------------- DATABASE -------------------
def init_db():
    conn = sqlite3.connect(DATABASE)
    conn.execute('''CREATE TABLE IF NOT EXISTS doctors
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     license_id TEXT UNIQUE,
                     name TEXT,
                     hospital TEXT,
                     city TEXT,
                     state TEXT,
                     password_hash TEXT)''')
    conn.commit()
    conn.close()

def add_doctor(license_id, name, hospital, city, state, password):
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    try:
        pw_hash = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)
        cur.execute("INSERT INTO doctors (license_id, name, hospital, city, state, password_hash) VALUES (?, ?, ?, ?, ?, ?)",
                    (license_id, name, hospital, city, state, pw_hash))
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()

def get_doctor(license_id):
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    cur.execute("SELECT * FROM doctors WHERE license_id=?", (license_id,))
    row = cur.fetchone()
    conn.close()
    return row

init_db()

# ------------------- LOAD MODELS -------------------
chest_model = tf.keras.models.load_model("chest_xray_model.keras")
mri_model = tf.keras.models.load_model("brain_mri_model.keras")

# ------------------- IMAGE UTILS -------------------
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_with_confidence(model, img_path):
    arr = preprocess_image(img_path)
    pred = model.predict(arr)[0][0]
    label = "Normal"
    confidence = round((1 - pred) * 100, 2)
    if pred >= 0.5:
        label = "Pneumonia" if model == chest_model else "Tumor"
        confidence = round(pred * 100, 2)
    return label, confidence

def make_gradcam(model, img_path, target_size=(224,224)):
    img_arr = preprocess_image(img_path, target_size)
    img = Image.open(img_path).convert("RGB").resize(target_size)
    last_conv = None
    for layer in reversed(model.layers):
        if "conv" in layer.name:
            last_conv = layer.name
            break
    grad_model = Model([model.inputs], [model.get_layer(last_conv).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_arr)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2)).numpy()
    conv_outputs = conv_outputs.numpy()[0]
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    cmap = plt.get_cmap("jet")
    heatmap = Image.fromarray(np.uint8(255 * cmap(heatmap)[:, :, :3]))
    blended = Image.blend(img, heatmap, alpha=0.45)
    return blended

# ------------------- ROUTES -------------------
@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        data = {k: request.form[k] for k in ["license_id", "name", "hospital", "city", "state", "password"]}
        if add_doctor(**data):
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for("login"))
        flash("License ID already exists!", "danger")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        license_id = request.form["license_id"]
        password = request.form["password"]
        doc = get_doctor(license_id)
        if doc and check_password_hash(doc[6], password):
            session["doctor_name"] = doc[2]
            session["doctor_id"] = doc[0]
            flash(f"Welcome Dr. {doc[2]}!", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid credentials.", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("landing"))

@app.route("/dashboard")
def dashboard():
    if "doctor_id" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", doctor_name=session.get("doctor_name"))

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "doctor_id" not in session:
        return redirect(url_for("login"))

    prediction, confidence, image_url, gradcam_url = None, None, None, None

    if request.method == "POST":
        dataset = request.form.get("dataset")
        file = request.files.get("file")

        if not (dataset and file and file.filename):
            flash("Please select scan type and upload an image.", "warning")
            return redirect(url_for("predict"))

        # Save image
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        # Select model and predict
        model = chest_model if dataset == "chest" else mri_model
        label, confidence = predict_with_confidence(model, save_path)
        prediction = label

        # Generate Grad-CAM
        try:
            grad_img = make_gradcam(model, save_path)
            grad_name = f"gradcam_{filename}.png"
            grad_path = os.path.join(app.config["UPLOAD_FOLDER"], grad_name)
            grad_img.save(grad_path)
            gradcam_url = url_for("uploaded_file", filename=grad_name)
        except Exception as e:
            print("Grad-CAM Error:", e)
            gradcam_url = None

        # Build correct URLs for rendering
        image_url = url_for("uploaded_file", filename=filename)

    return render_template("predict.html",
                           prediction=prediction,
                           confidence=confidence,
                           image_url=image_url,
                           gradcam_url=gradcam_url)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_file(os.path.join(app.config["UPLOAD_FOLDER"], filename))

if __name__ == "__main__":
    app.run(debug=True)
