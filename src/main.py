from reid_pipeline import Pipeline, DetectedObject


from server.milvus_helpers import MilvusHelper
from server.mysql_helpers import MySQLHelper
from server.server_pipeline import ServerPipeline
from server.surveillance_camera_manager import SurveillanceCameraManager
from server.query_system_backend import QuerySystemBackend

import numpy as np
import cv2

import glob
import time

import argparse

def get_parser():
    parser = argparse.ArgumentParser("Bicycle ReID")
    parser.add_argument(
        "-mode", default="q", help="q:query system, s:surveillance camera"
    )
    parser.add_argument("-cam_id", default=0, help="camera id")
    parser.add_argument("-cam_url", default=0, help="camera url")
    return parser

def main(args):
    server_pipeline = ServerPipeline()
    if args.mode == "q":
        backend = QuerySystemBackend(server_pipeline)
        backend.launch()
    if args.mode == "s":
        manager = SurveillanceCameraManager(server_pipeline, cam_id=args.cam_id, camera_url=args.cam_url)
        manager.run()



if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
    
