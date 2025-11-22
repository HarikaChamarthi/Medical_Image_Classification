# Medical Image Classification  
*Deep-learning solution to classify medical images into healthy vs. diseased categories.*

## 🔍 Project Overview  
This project develops a convolutional neural network (CNN)-based classification system that takes medical images (e.g., X-ray, MRI, CT) as input and outputs predictions whether the image is **healthy** or **diseased**.  
It is implemented in Python using popular libraries (e.g., TensorFlow / PyTorch, NumPy, etc.) and structured for ease of extension and deployment.

## 📁 Project Structure  
Medical_Image_Classification/

├─ uploads/ # raw image uploads or sample dataset (if included)

├─ app.py # main application script (inference / web interface)

├─ app.spec # specification for packaging (if used)

├─ dashboard.html # (optional) web UI dashboard for monitoring / visualization

├─ landing.html # homepage for the web interface

├─ login.html # user login page (if access control)

├─ predict.html # front-end page to upload image and view prediction

├─ register.html # user registration page (if applicable)

└─ doctors.db # database for user/patient or model logging (if applicable)



🎯 Key Features
------------------------------------------------------------------------------------------------------

CNN architecture optimized for medical image classification.

User-friendly front-end (HTML pages) for image upload + real-time inference.

Database logging of user uploads/predictions for audit / tracking (via doctors.db).

Modular code—easy to replace model, dataset or add more classes.


📦 Libraries Used
----------------------------------------------------------------------------------------

This project uses the following major Python libraries:

TensorFlow / Keras – for building and training the CNN model

NumPy – for numerical operations

scikit-learn – for data splitting, evaluation metrics, and preprocessing

OpenCV (opencv-python) – for image loading and resizing

Flask – to create the web interface for uploading and classifying images

Matplotlib – for visualizing training accuracy and loss

SQLite3 – for storing user/doctor information in doctors.db



🧑‍💻 Author
--------------------------------------------------------------------------------

Harika Chamarthi

Contact or check out my GitHub profile for more projects.
