from server.config import *
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

import numpy as np
import torch

class MilvusHelper:
    def __init__(self, collection_name):
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        self.collection = None
        self.collection_name = collection_name
        self.create_collection()
        self.create_index()

    def get_num_entities(self):
        self.set_collection()
        self.collection.flush()
        num = self.collection.num_entities
        return num

    def set_collection(self):
        if self.has_collection():
            self.collection = Collection(name=self.collection_name)
  
    def has_collection(self):
        return utility.has_collection(self.collection_name)


    def create_collection(self):
        # Create milvus collection if not exists
        if not self.has_collection():
                
            bicycle_id = FieldSchema(name="bicycle_id",
                                    dtype=DataType.INT64,
                                    descrition="the only id for every bike-person pair",
                                    is_primary=True,
                                    auto_id=True)
                
            bicycle_embedding = FieldSchema(name="bicycle_embedding", 
                                            dtype=DataType.FLOAT_VECTOR, 
                                            descrition="float bicycle_embedding",
                                            dim=VECTOR_DIMENSION,
                                            is_primary=False,
                                            auto_id = False)
                
            schema = CollectionSchema(fields = [bicycle_id,
                                                bicycle_embedding], 
                                                description="bicycle search")
                
            self.collection = Collection(name=self.collection_name, schema=schema)
        else:
            self.set_collection()
            return "OK"
        
    def create_index(self):
        # Create IVF_FLAT index on milvus collection
        self.set_collection()
        if self.collection.has_index():
            return None
        default_index = {"index_type": INDEX_TYPE, "metric_type": METRIC_TYPE, "params": {"nlist": 16384}}
        # * nlist:16384后续可注意是否修改
        status = self.collection.create_index(field_name="bicycle_embedding", index_params=default_index, timeout=60)
        return status
        

    def search_vectors(self, vectors, top_k):
        # Search vector in milvus collection
        for i, vector in enumerate(vectors):
            if isinstance(vector, torch.Tensor):
                vectors[i] = vector.detach().numpy()
            if isinstance(vector, np.ndarray):
                vectors[i] = vector.astype(np.float32)
        self.set_collection()
        self.collection.load()
        search_params = {"metric_type": METRIC_TYPE, "params": {"nprobe": 16}}
        res = self.collection.search(vectors, anns_field="bicycle_embedding", param=search_params, limit=top_k)
        if len(res[0].distances) > 0:
            print("distances:",res[0].distances[0])
        return res 
        # res[0].ids:get the IDs of all returned hits
        # res[0].distances:get the distances to the query vector from all returned hits

    def insert_new_bike(self, bicycle_embedding):
        # 自行车embedding逐条插入数据库，先在milvus里面检索top1,如果与top1的相似度大于阈值，就不插入，否则插入
        if isinstance(bicycle_embedding, torch.Tensor):
            bicycle_embedding = bicycle_embedding.detach().numpy()

        self.set_collection()
        self.collection.load()
        search_result = self.search_vectors([bicycle_embedding], top_k=1)
            
        if isinstance(bicycle_embedding, np.ndarray):
            bicycle_embedding = bicycle_embedding.astype(np.float32)
            if(len(search_result[0].distances) > 0 and search_result[0].distances[0] > DISTANCE_THERSHOLD):
                return search_result[0].ids[0]
            else:
                mr = self.collection.insert([[bicycle_embedding]])
                id = mr.primary_keys[0]
                return id
            
        else:
            raise TypeError("bicycle_embedding type error, expect numpy.ndarray or torch.Tensor, got %s"%type(bicycle_embedding))

    def check_bicycle_exist(self, bicycle_embedding):
        if isinstance(bicycle_embedding, torch.Tensor):
            bicycle_embedding = bicycle_embedding.detach().numpy()

        self.set_collection()
        self.collection.load()
        search_result = self.search_vectors([bicycle_embedding], top_k=1)
            
        if isinstance(bicycle_embedding, np.ndarray):
            bicycle_embedding = bicycle_embedding.astype(np.float32)
            if(len(search_result[0].distances) > 0 and search_result[0].distances[0] > DISTANCE_THERSHOLD):
                return search_result[0].ids[0]
            else:
                return -1           
        else:
            raise TypeError("bicycle_embedding type error, expect numpy.ndarray or torch.Tensor, got %s"%type(bicycle_embedding))