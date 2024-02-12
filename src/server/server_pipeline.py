from server.config import *
from reid_pipeline import Pipeline, DetectedObject
from server.milvus_helpers import MilvusHelper
from server.mysql_helpers import MySQLHelper

import random

from datetime import datetime

# *? 该文件实现数据库的创建，插入，搜索，按时删除等功能
# |   0   |     1      |     2     |     3      |    4     |       5       |    6     |
# | :---: | :--------: | :-------: | :--------: | :------: | :-----------: | :------: |
# |  id   | bicycle_id | camera_id | start_time | end_time | location_desc | img_path |

class ServerPipeline:
    def __init__(self) -> None:
        self.pipeline = Pipeline()
        self.milvus = MilvusHelper(MILVUS_DB_NAME)
        self.mysql = MySQLHelper()
        self.mysql.create_mysql_table(MYSQL_TABLE)
    
    def insert_new_data_from_img(self, img, cam_id):
    # 主功能一：将监控图像中的自行车插入数据库
        if random.random() > 0.9999:
            self.mysql.auto_delete_ExpiredData(MYSQL_TABLE, delete_interval=MYSQL_DELETE_INTERVAL)
        objs = self.pipeline(img, cam_id=cam_id)
        for obj in objs:
            if obj.cls_id == 1:  
                bike_id = self.milvus.insert_new_bike(obj.embedding)
                self.insert_bike_occurrence(bike_id, obj)

    def query_img(self, img, top_k=10):
    # 主功能二：接受用户查询，返回前top_k辆相似的自行车，并返回自行车出现的记录
        objs = self.pipeline(img, cam_id=-1)
        ui_content = []
        if objs is not None and len(objs) > 0:
            objs = [obj for obj in objs if obj.cls_id == 1]
            obj = max(objs, key=lambda x: x.score*x.img.shape[0]*x.img.shape[1])
            bike_occurrence_res = self.get_bike_occurrence(obj, top_k)
            # print("len of bike_occurrence_res: ", len(bike_occurrence_res))
            for item in bike_occurrence_res: # list of (bike_id, res)
                item[1] = self.mysql.search_result_to_df(item[1])
                if item[1].shape[0] > 0:
                    ui_content.append(item[1]["img_path"][0])
                    df = item[1].drop(["img_path"], axis=1)
                    df["start_time"] = df["start_time"].apply(lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S"))
                    df["end_time"] = df["end_time"].apply(lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S"))
                    ui_content.append(df)

        if len(ui_content) < 20:
            for i in range(20-len(ui_content)):
                if i % 2 == 0:
                    ui_content.append("server/src/no_result_found.png")
                else:
                    ui_content.append(None)
        
        return ui_content

    

    def insert_bike_occurrence(self, bike_id, obj: DetectedObject):
        bike_occurrence_res = self.get_bike_occurrence(obj, top_k=1)
        single_bike_occurrence_res = None
        for res in bike_occurrence_res:
            single_bike_occurrence_res = res[1]
        if single_bike_occurrence_res is not None and len(single_bike_occurrence_res)>0:
            single_bike_occurrence_dict = self.mysql.db_line_to_dict(single_bike_occurrence_res[0])
            if obj.cam_id == single_bike_occurrence_dict["camera_id"]:
                self.mysql.update_end_time(MYSQL_TABLE, single_bike_occurrence_dict["id"], obj.time)
            else:
                self.mysql.insert(MYSQL_TABLE, bike_id, obj)
        else:
            self.mysql.insert(MYSQL_TABLE, bike_id, obj)
        

    def get_bike_occurrence(self, obj, top_k=10):
        bike_occurrence_res = []

        if obj.cls_id == 1: 
            search_result = self.milvus.search_vectors([obj.embedding], top_k)
            for res in search_result:
                for i in range(len(res.ids)):
                    bike_id = res.ids[i]
                    bike_occurrence_res.append([bike_id,self.mysql.search_by_bicycle_id(MYSQL_TABLE,bike_id)])

        return bike_occurrence_res
    
    def bike_occurrence_res_to_str(self, bike_occurrence_res:dict):
        for single_bike_occurrence_res in bike_occurrence_res.values():
            for db_line in single_bike_occurrence_res:
                self.mysql.db_line_to_str(db_line)
    
    def count_bike(self):
        if not self.milvus.has_collection():
            return 0
        return self.milvus.get_num_entities()



