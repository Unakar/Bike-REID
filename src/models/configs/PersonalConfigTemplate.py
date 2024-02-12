class PersonalConfig:
    def __init__(self):
        self.bike_person_dataset_dir = '/home/aistudio/data/data237899/BikeDataset'
        self.yolox_exp_file = 'models/configs/yolox_exps/default/yolox_m.py'
        self.yolox_ckpt = 'models/weights/yolox_m.pth'
        self.yolox_path = "/home/aistudio/data/data237899/cam_1_2/Bike/Person_0000"
    def get_bike_person_dataset_dir(self):
        return self.bike_person_dataset_dir