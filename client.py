from flask import Flask, render_template, Response
import cv2
import threading
from v5_dnn import *
from goto import with_goto

app = Flask(__name__)

source = "rtsp://admin:anas1155@46.152.196.211:554/Streaming/Channels/1/"
yolonet = yolov5()



@with_goto
def video_stream():
    label.begin
    cap = cv2.VideoCapture(source)
    while True:

        ret, frame = cap.read()
        if ret == False :
            print('there is problem')
            goto.begin
        yolonet.v5_inference(frame)
        #re = cv2.resize(frame, (700, 500))
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
