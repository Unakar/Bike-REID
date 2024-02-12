import pymysql
import cv2
import pandas as pd

import time


from server.config import *
from datetime import datetime, timedelta
from reid_pipeline.reid_data_manager import DetectedObject



class MySQLHelper:
    def __init__(self):
        self.conn = pymysql.connect(host=MYSQL_HOST,
                                    user=MYSQL_USER,
                                    port=MYSQL_PORT,
                                    password=MYSQL_PWD,
                                    database=None,
                                    local_infile=True)
        
        self.cursor = self.conn.cursor()
        self.cursor.execute("CREATE DATABASE IF NOT EXISTS "+MYSQL_DB+" DEFAULT CHARACTER SET utf8 COLLATE utf8_general_ci;")
        self.cursor.execute("USE "+MYSQL_DB+";")

    def test_connection(self):
        try:
            self.conn.ping()
        except Exception:
            self.conn = pymysql.connect(host=MYSQL_HOST, user=MYSQL_USER, port=MYSQL_PORT, password=MYSQL_PWD,
                                        database=MYSQL_DB, local_infile=True)
            self.cursor = self.conn.cursor()

    def create_mysql_table(self, table_name):
        # Create mysql table if not exists
        self.test_connection()
        sql = "CREATE TABLE IF NOT EXISTS "+table_name +" "+\
        "(id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT, "+\
        "bicycle_id BIGINT UNSIGNED DEFAULT NULL, "+\
        "camera_id INT UNSIGNED DEFAULT NULL, "+\
        "start_time BIGINT DEFAULT NULL, "+\
        "end_time BIGINT DEFAULT NULL, " +\
        "location_desc VARCHAR(50) DEFAULT NULL, " +\
        "img_path VARCHAR(100) DEFAULT NULL, " +\
        "PRIMARY KEY ( `id` ), KEY `index_bicycle_id` ( `bicycle_id` ) USING BTREE "+\
        ") ENGINE = INNODB DEFAULT CHARSET = utf8;"

        self.cursor.execute(sql)
        self.conn.commit()

    def insert(self, table_name, bike_id, obj: DetectedObject):
        # 单条数据插入，返回最后一行id
        self.test_connection()
        # time_str = "%s"%datetime.fromtimestamp(obj.time)

        # save_img
        save_folder = SAVE_IMG_PATH
        os.makedirs(save_folder, exist_ok=True)
        img_path = os.path.join(
            save_folder, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        )
        img_path = img_path + "_bike_%d"%bike_id + ".jpg"
        cv2.imwrite(img_path, obj.bike_person_img)
        
        sql = "INSERT INTO " + table_name + " "\
            "(bicycle_id, camera_id, start_time, end_time, location_desc, img_path) "+\
            "VALUES (%d, %d, %ld, %ld, '%s', '%s');" % (bike_id, obj.cam_id, int(obj.time), int(obj.time), str(obj.cam_id), img_path)
        n = self.cursor.execute(sql)
        if n > 0:
            ms_id = self.cursor.lastrowid 
            self.conn.commit()
        else:
            self.conn.rollback()
        return ms_id


    def search_by_bicycle_id(self, table_name, bike_id):
        self.test_connection()
        sql = "SELECT * from " + \
            table_name + " where bicycle_id in (%d) ORDER BY end_time DESC;" % bike_id
        self.cursor.execute(sql)
        results = self.cursor.fetchall() # results类型为嵌套的tuple 
        return results

    def search_by_bicycle_ids(self, table_name, bike_ids):
        self.test_connection()
        str_bike_ids = str(list(bike_ids)).replace('[', '').replace(']', '')
        sql = "select * from " + \
            table_name + " where bicycle_id in (" + str_bike_ids + ") ORDER BY end_time DESC;"
        self.cursor.execute(sql)
        results = self.cursor.fetchall()
        print(results)
        return results


    def delete_by_id(self, table_name, id):
        self.test_connection()
        sql = "DELETE FROM  %s WHERE id = %s;" % (table_name, id)
        self.cursor.execute(sql)
        self.conn.commit()


    def update_end_time(self, table_name, id, end_time):
        # end_time is a timestamp(sec)
        self.test_connection()
        sql = "UPDATE " + table_name + " " +\
            "SET end_time=%ld "%(int(end_time)) +\
            "WHERE id=%d;"%id
        n = self.cursor.execute(sql)
        self.conn.commit()


    def auto_delete_ExpiredData(self,table_name, delete_interval=MYSQL_DELETE_INTERVAL):
        self.test_connection()
        # 计算7天前的时间
        time_limit = datetime.now() - timedelta(days=delete_interval)
        sql = f"DELETE FROM " + table_name+" WHERE end_time < %ld"%(int(time_limit.timestamp()))
        self.cursor.execute(sql)
        self.conn.commit()        
        rows_deleted = self.cursor.rowcount
        self.cursor.close()
        return rows_deleted#返回受影响的列

    def db_line_to_str(self, res):
        # res: tuple
        # return: string
        desc = ""
        desc += "bicycle_id: %d\n"%res[1]
        desc += "camera_id: %d\n"%res[2]
        desc += "start_time: %s\n"%datetime.fromtimestamp(res[3])
        desc += "end_time: %s\n"%datetime.fromtimestamp(res[4])
        desc += "location_desc: %s\n"%res[5]
        desc += "img_path: %s\n"%res[6]
        return res
    
    def db_line_to_dict(self, res):
        # res: tuple
        # return: dict
        desc = {}
        desc["id"] = res[0]
        desc["bicycle_id"] = res[1]
        desc["camera_id"] = res[2]
        desc["start_time"] = datetime.fromtimestamp(res[3])
        desc["end_time"] = datetime.fromtimestamp(res[4])
        desc["location_desc"] = res[5]
        desc["img_path"] = res[6]
        return desc
    
    def search_result_to_df(self,res):
        # res: tuple
        # return: dataframe
        df = pd.DataFrame(list(res), columns=["id", "bicycle_id", "camera_id", "start_time", "end_time", "location_desc", "img_path"])
        return df
