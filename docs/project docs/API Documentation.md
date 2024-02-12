# API文档

## 项目结构

- run: 启动整个程序运行的逻辑，包括通过服务器提供服务、训练、测试、评估等
- reid_pipeline: Re-id流程主体逻辑
- models: 模型具体实现，包括各模型和用于管理模型的类
- database_manager: 数据库管理，包括数据增删查改等
- utils: 工具类，包括各种工具函数、工具类


## 基本API

### run


### reid_pipeline
#### pipeline.py
```python
class Pipeline:
    def __init__(self):
        pass
    def __call__(self, *args, **kwargs):
        pass
    def spot_object_from_image(self, image):
        pass
    def spot_object_from_video(self, video):
        pass
    def get_embedding(self, objects):
        pass
    def submit_result(self, embeddings):
        pass

```

#### reid_data_manager.py
```python
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


```

## 数据库结构

#### sever_pipeline.py
```python
class ServerPipeline:
    def insert_new_data_from_img(self, img, cam_id):
    # 主功能一：将监控图像中的自行车插入数据库

    def query_img(self, img, top_k=10):
    # 主功能二：接受用户查询，返回前top_k辆相似的自行车，并返回自行车出现的记录

```
### MySQL

|   0   |     1      |     2     |     3      |    4     |       5       |    6     |
| :---: | :--------: | :-------: | :--------: | :------: | :-----------: | :------: |
|  id   | bicycle_id | camera_id | start_time | end_time | location_desc | img_path |



|     Field     |        Type         | Null  |  Key  | Default |     Extra      |
| :-----------: | :-----------------: | :---: | :---: | :-----: | :------------: |
|      id       | bigint(20) unsigned |  NO   |  PRI  |  NULL   | auto_increment |
|  bicycle_id   | bigint(20) unsigned |  YES  |  MUL  |  NULL   |                |
|   camera_id   |  int(10) unsigned   |  YES  |       |  NULL   |                |
|  start_time   |     bigint(20)      |  YES  |       |  NULL   |                |
|   end_time    |     bigint(20)      |  YES  |       |  NULL   |                |
| location_desc |     varchar(50)     |  YES  |       |  NULL   |                |
|   img_path    |    varchar(100)     |  YES  |       |  NULL   |                |


### milvus

|     0      |     1    |
| :--------: | :-------: |
| bicycle_id | embedding |
