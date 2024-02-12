from server.server_pipeline import ServerPipeline

import cv2

import glob

def get_imgs(path_pattern):
    imgs=[]
    img_paths = glob.glob(path_pattern)
    for img_path in img_paths:
        imgs.append(cv2.imread(img_path))
    return imgs


class SurveillanceCameraManager:
    def __init__(self, server_pipeline: ServerPipeline, cam_id, camera_url=0):
        # camera_url为0表示电脑内置摄像头，为文件路径或远程摄像头地址（http service）则表示外接摄像头
        self.server_pipeline = server_pipeline
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(camera_url)

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                self.server_pipeline.insert_new_data_from_img(frame, self.cam_id)
            else:
                break

    def read_img(self, path):
        imgs = get_imgs(path)
        for img in imgs:
            self.server_pipeline.insert_new_data_from_img(img, self.cam_id)
