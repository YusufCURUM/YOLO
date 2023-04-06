from flask import Flask, render_template, Response
import argparse
from v3_fastest import *
from v4_tiny import *
from v5_dnn import *
from vx_ort import *
from goto import with_goto
from goto import goto, label

class VideoCamera(object):


    def __init__(self):
        # opencv
        source1 = 0
        source2 = "rtsp://admin:anas1155@46.152.196.211:554/Streaming/Channels/1?tcp/"
        source3 = "cctv.mp4"
        self.video = cv2.VideoCapture(source2)

    def __del__(self):
        self.video.release()
    def get_frame(self):
        success, image = self.video.read()
        # opencv jpeg motion JPEG jpg
        # ret, jpeg = cv2.imencode('.jpg', image)
        # return jpeg.tobytes()
        return image

app = Flask(__name__)

@app.route('/')
def index():
    # jinja
    return render_template('index.html')


def v3_fastest(camera):
    while True:
        frame = camera.get_frame()
        v3_inference(frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        # generator，content image/jpeg, enimcodo base64
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def v4_tiny(camera):
    while True:
        frame = camera.get_frame()
        v4_inference(frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



def v5_dnn(camera):

    v5_net = yolov5()
    while True:

        frame = camera.get_frame()


        v5_net.v5_inference(frame)

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



def vx_ort(camera):
    while True:
        frame = camera.get_frame()
        yolox_detect(frame)
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():

    return Response(v5_dnn(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object Detection using YOLO-Fastest in OPENCV')
    parser.add_argument('--model', type=str, default='vx_ort', choices=['v3_fastest', 'v4_tiny', 'v5_dnn', 'vx_ort'])
    parser.add_argument('--semi-label', type=int, default=0, help="semi-label the frame or not")
    args = parser.parse_args()
    model = args.model
    app.run(host='0.0.0.0', debug=True, port=5000)
