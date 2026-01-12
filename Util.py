# encoding = utf-8
import cv2

def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(fps)

if __name__ == '__main__':
    get_fps(video_path='./videos/2025-06-11 10-27-50.mkv')