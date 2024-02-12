import torch
import time

class DetectedObject: #封装检测到的目标的数据
    def __init__(self, cam_id, img, bike_person_img, score, cls_id, center):
        self.cam_id = cam_id
        self.img = img
        self.bike_person_img = bike_person_img
        self.score = score
        self.cls_id = cls_id
        self.center = center
        self.embedding: torch.Tensor = None
        self.time = time.time() #时间戳除以1000，单位为秒

    def __str__(self):
        info = "img.size(height, width, channel) = "+str(self.img.shape)+" center(x, y) = "+str(self.center)+"\n"
        info += "score = %.2f cls_id = %d\n"%(self.score, self.cls_id)
        if self.embedding is not None:
            info += "embedding: "+str(self.embedding)+"  embedding.shape = "+str(self.embedding.shape)
        return info