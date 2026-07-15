# fetch_face_models.py
"""Download the OpenCV YuNet + SFace models into models/ (gitignored).

Run once after cloning:  python fetch_face_models.py
"""
import os
import urllib.request

MODELS = {
    "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx",
}

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    dest = os.path.join(here, "models")
    os.makedirs(dest, exist_ok=True)
    for name, url in MODELS.items():
        path = os.path.join(dest, name)
        if os.path.isfile(path):
            print(f"[skip] {name} already present")
            continue
        print(f"[get ] {name} ...")
        urllib.request.urlretrieve(url, path)
        print(f"[ok  ] {name} ({os.path.getsize(path)//1024} KB)")

if __name__ == "__main__":
    main()
