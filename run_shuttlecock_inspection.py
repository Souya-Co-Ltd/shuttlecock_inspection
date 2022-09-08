"""
script for visual inspection of shuttlecock
"""
import argparse
from pathlib import Path
from typing import Tuple
from time import perf_counter

import numpy as np
import tensorflow as tf
import cv2


class ShuttlecockInspector:
    """
    ShuttlecockInspector
    """

    labels = ["defective", "good"]

    def __init__(self, model_filepath: Path):
        """
        model_filepath: model filepath
        """
        self.model_filepath: Path = model_filepath

        # load model
        self.interpreter = tf.lite.Interpreter(model_path=str(model_filepath))
        self.interpreter.allocate_tensors()

        # get inout/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


    def predict(self, image: np.ndarray) -> Tuple[float, str]:
        """
        Predict
        Args:
            image: target image
        """
        normalized_image = (image.astype(np.float32) / 127.0) - 1
        data = np.asarray([normalized_image], dtype=np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], data)
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
        # prediction = self.model(data, training=False)
        max_index = np.argmax(prediction)
        return float(prediction[0][max_index]), self.labels[max_index]



def main(model_filepath: Path, device: str):
    # create inspector
    inspector = ShuttlecockInspector(model_filepath)

    # =====================================================
    # Capture
    # =====================================================
    # capture pipeline
    input_width = 640
    input_height = 480
    framerate = 30
    capture_pipeline = f"v4l2src device={device} \
        ! video/x-raw, width=(int){input_width}, height=(int){input_height}, framerate={framerate}/1, format=(string)YUY2 \
        ! videoconvert ! video/x-raw, format=BGR \
        ! appsink max-buffers=1 drop=True"

    # create capture
    capture = cv2.VideoCapture()
    try:
        # capture open
        capture.setExceptionMode(True)
        capture.open(capture_pipeline, cv2.CAP_GSTREAMER)
        if not capture.isOpened():
            print(f"{device} is can't opened")
            return

    except Exception as e:
        print(f"{device} is can't opened")
        print("Camera Error", e)
        if capture.isOpened():
            capture.release()
        return

    # =====================================================
    # Writer
    # =====================================================
    # write pipeline
    write_pipeline = f"appsrc ! nveglglessink sync=False"
    # write_pipeline = "appsrc ! video/x-raw, format=BGR ! videoconvert ! x264enc ! flvmux ! filesink location=xyz.flv"

    # create writer
    writer = cv2.VideoWriter()
    try:
        # writer open
        writer.open(write_pipeline, cv2.CAP_GSTREAMER, 0, framerate, (input_width, input_height))
        if not writer.isOpened():
            print(f"writer is can't opend")
            return

    except Exception as e:
        print(f"writer is can't opened")
        print("Camera Error", e)
        if writer.isOpened():
            writer.release()
        return

    # =====================================================
    # capture, predict, write
    # =====================================================
    width = 224
    height = 224
    min_length = min(input_width, input_height)
    top = (input_height - min_length) // 2
    bottom = input_height if top == 0 else -top
    left = (input_width - min_length) // 2
    right = input_width if left == 0 else -left
    try:
        lap_time = perf_counter()
        while True:
            # capture
            result, frame = capture.read()
            if not result:
                continue

            # crop
            frame_for_predict = frame[top:bottom, left:right, :]
            frame_for_predict = cv2.resize(frame_for_predict, (height, width))

            # predict
            accuracy, label = inspector.predict(frame_for_predict)

            # write predict result
            fps = 1.0 / (perf_counter() - lap_time)
            fps = framerate if fps > framerate else fps
            cv2.putText(frame, f"{label} - {accuracy:.3f} - {fps:.1f}fps", (0, 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 1, cv2.LINE_AA)

            # display
            writer.write(frame)
            lap_time = perf_counter()

    except KeyboardInterrupt:
        pass

    except Exception as e:
        print("Error", e)

    finally:
        if capture.isOpened():
            capture.release()
        if writer.isOpened():
            writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visual inspection of shuttlecock")
    parser.add_argument("-d", "--device", default="/dev/video0", help="video device")
    args = parser.parse_args()

    model_filepath = Path("./model/model_unquant.tflite")
    device = args.device
    main(model_filepath, device)
