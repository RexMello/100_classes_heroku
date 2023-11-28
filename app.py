from flask import Flask, request, jsonify, send_file
import cv2
import os
import pandas as pd
import csv
import smtplib
from yolov8 import YOLOv8
from flask_cors import CORS
import gdown

app = Flask(__name__)
CORS(app)

try:
    os.mkdir('models')
except:
    pass

if not os.path.exists('models/logos.onnx'):
    url = 'https://drive.google.com/uc?id=1fubmQ0I6D1Ov5smU6MoWGq43OX7mbdZy'
    output = 'models/logos.onnx'
    gdown.download(url, output, quiet=False)

if not os.path.exists('models/objects.onnx'):
    url = 'https://drive.google.com/uc?id=1AKPsU8DpFUReJLUj0p7oJ71aJ-mDmtHL'
    output = 'models/objects.onnx'
    gdown.download(url, output, quiet=False)

def look_for_detections(name, df):
    # Convert DataFrame to a dictionary
    result_dict = df.to_dict(orient='records')

    # Find the email associated with the 'Name' value 'default'
    times = ''

    time = []

    for record in result_dict:
        if record['Name'] == name and record['Time'] not in time:
            time.append(record['Time'])
            times+=', '+str(record['Time'])

    return times[2:]

def run_processing(cap, draw, email, yolov8_detector, class_names):
    # Create VideoWriter object to save the output video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 file format
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    detected_objects = []

    # ------------------ Detecting goal and shoot events ---------------------
    while True:
        ret,frame = cap.read()

        if not ret:
            print('Video ended')
            break

        sec = int(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)

        boxes, scores, class_ids = yolov8_detector(frame)
        for box, _, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            object_name = class_names[class_id]

            blur_roi = frame[y1:y2,x1:x2]
            blur_roi = cv2.GaussianBlur(blur_roi, (25, 25), 0)
            frame[y1:y2,x1:x2] = blur_roi
            
            
            if object_name not in detected_objects:
                detected_objects.append(object_name)
            
            save_to_csv(object_name,sec)

            if draw:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.putText(frame,object_name,(x1,y1-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1)

        out.write(frame)
    # ------------------------------------------------------------------------
    
    cap.release()
    out.release()

    print("################# SENDING EMAIL #################")
    send_mails(detected_objects, email)
    print("################# EMAIL SENT #################")

def save_to_csv(name, time):
    file_path = 'output.csv'

    is_new_file = not os.path.exists(file_path)
    with open(file_path, mode='a', newline='') as csvfile:
        fieldnames = ['Name', 'Time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if is_new_file:
            writer.writeheader()

        # Write the data to the CSV file.
        writer.writerow({'Name': name, 'Time': time})

def send_mails(detections, email_receiver):
    # Define email sender and receiver
    try:
        det = pd.read_csv('output.csv')
    except:
        print('NOTHING TO EMAIL')
        return
    email_sender = 'kai.official.003@gmail.com'
    email_password = 'tazj jnnw avfb eeif'

    messages = ''
    for detection in detections:
        times_of_detection = look_for_detections(detection,det)
        messages+=f'\nObjects: {str(detection)}\nTimes: {times_of_detection}'
        
    server = smtplib.SMTP('smtp.gmail.com',587)
    server.starttls()
    server.login(email_sender, email_password)
    server.sendmail(email_sender,email_receiver,messages)


@app.route('/api/run_detections', methods=['POST'])
def api_run_detections():
    print("################# DELETING STUFF #################")
    try:    
        os.remove('temp_video.mp4')
    except:
        pass

    try:    
        os.remove('output.mp4')
    except:
        pass

    try:    
        os.remove('output.csv')
    except:
        pass
    

    print("################# STARTING PROCESS #################")
    # Get the video file from the request
    video_file = request.files.get('video')

    if video_file is None:
        return jsonify({'error': 'No video file provided'})

    # Save the video file to a temporary location
    video_path = 'temp_video.mp4'
    video_file.save(video_path)

    # Read the video file using OpenCV
    cap = cv2.VideoCapture(video_path)

    email = request.form.get('email')
    draw = request.form.get('draw', True)
    model = request.form.get('model')
    confidence = request.form.get('confidence')


    if email is None or email=='':
        return jsonify({'Error':'Email not found'})

    if model is None or str(model)+'.onnx' not in os.listdir('models'):
        return jsonify({'Error':'Model name not found'})

    try:
        confidence = int(confidence)
    except:
        return jsonify({'Error':'Invalid confidence'})

    try:
        if str(draw).lower() == 'true':
            draw = True
        else:
            draw = False
    except:
        return jsonify({'Error':'Unknown draw value'})


    if model == 'objects':
        #--------------------- Initialize object detectors for flood detection --------------------------
        model_path = f"models/objects.onnx"
        yolov8_detector = YOLOv8(model_path, conf_thres=(confidence-10)/100, iou_thres=confidence/100)
        class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccobroccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        #------------------------------------------------------------------------------------------------
    elif model == 'logos':
        #--------------------- Initialize object detectors for flood detection --------------------------
        model_path = f"models/logos.onnx"
        yolov8_detector = YOLOv8(model_path, conf_thres=confidence-10/100, iou_thres=confidence/100)
        class_names = ['Ducati', 'Jacques Marie Mage glasses', 'Jansport', 'Jeep', 'Kawasaki', 'Kia', 'Lacoste', 'LandRover', 'Lego', 'Levis', 'Lexus', 'LG', 'Lincoln', 'Louis Vuitton', 'Maserati', 'Mercedes Benz', 'Microsoft', 'Mitsubishi', 'Moscot', 'NBA', 'Nike', 'Nissan', 'Now This', 'Oakley', 'Omega', 'Osprey', 'Panavision', 'Persol', 'Philadelphia 76ers', 'Porsche', 'Ray Ban', 'Samsung', 'Shoei', 'Shure', 'Sitka', 'Sony', 'Stella Artois', 'Subaru', 'TAG Heuer', 'The Scorpion King', 'Thomas the Tank Engine', 'Toyota', 'Tumi', 'Under Armour', 'UNIF', 'Volkswagen', 'Volvo', 'Wilson', 'Yamaha', 'YouTube', 'adidas', 'air jordan', 'apple', 'asus', 'audi', 'bajaj', 'bmw', 'bose', 'budweiser', 'c. f. martin & company', 'cadillac', 'carhartt', 'champion', 'chevrolet', 'chrome Industries', 'cnn', 'coca-cola', 'converse', 'corona', 'cutler and gross', 'david clark', 'dell', 'dodge', 'energica', 'federal donuts', 'ferrari', 'fiji water', 'ford', 'fossil', 'frys electronics', 'garrett leight', 'gibson', 'gmc', 'google', 'goorin brothers', 'gucci', 'hamilton', 'honda', 'hummer', 'hyundai', 'icee', 'instagram', 'modelo', 'monster', 'prime energy', 'puma', 'shure', 'skoda', 'the north face', 'versace']
        #------------------------------------------------------------------------------------------------
    else:
        return jsonify({'Error':'Invalid model name'})

    print("################# STARTING DETECTION #################")
    # Save processed video in memory
    run_processing(cap, draw, email, yolov8_detector, class_names)
    print("################# PROCESS COMPLETE #################")
    cap.release()

    return send_file('output.mp4', as_attachment=True)


@app.route('/', methods=['GET'])
def api_test():
    return jsonify({'status': 'API is active'})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)