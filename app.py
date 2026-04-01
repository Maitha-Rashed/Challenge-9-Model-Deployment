#Importing the libraries:
import os
import cv2
from flask import Flask, render_template, request
from ultralytics import YOLO

#Creating the Flask web application:
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"  #Folder where uploaded and predicted images will be stored.
MODEL_PATH = "model/best.pt"  #Path to the trained YOLO model file.

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER  #Storing the upload folder path in the Flask app settings.

model = YOLO(MODEL_PATH)  #Loading the trained YOLO model once when the app starts.

os.makedirs(UPLOAD_FOLDER, exist_ok=True)  #Creating the upload folder if it does not already exist.

#Defining the home page route:
@app.route("/")
def home():
    return render_template("index.html")  #Showing the main HTML page.

#Defining the route that handles image upload and prediction:
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:  #Checking if an image file was sent from the form.
        return "No image uploaded"  #Returning this message if no image was uploaded.

    file = request.files["image"]  #Getting the uploaded image file from the form.

    if file.filename == "":  #Checking if the user selected a file.
        return "No selected file"  #Returning this message if no file was chosen.

    #Saving the uploaded image:
    input_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)  #Creating the full path for the uploaded image.
    file.save(input_path)  #Saving the uploaded image in the uploads folder.

    #Running YOLOv8 prediction on the uploaded image:
    results = model.predict(source=input_path, conf=0.5) #Only keep detections with confidence of at least 0.5.

    plotted_image = results[0].plot()  #Drawing bounding boxes, labels, and confidence scores on the image.

    #Saving predicted image inside static/uploads folders:
    output_filename = "pred_" + file.filename  #Creating a new filename for the predicted image.
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)  #Creating the full path for the predicted image.
    cv2.imwrite(output_path, plotted_image)

    #Fixing slashes for browser display:
    uploaded_image = input_path.replace("\\", "/")
    output_image = output_path.replace("\\", "/")

    #Sending both images to the HTML page so it can be displayed:
    return render_template("index.html", uploaded_image=uploaded_image, output_image=output_image)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)