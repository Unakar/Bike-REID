# YOLO Quick Start
## 安装YOLO
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple ultralytics

建议安装：
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple opencv-python

## 最简单的实例
```python
from ultralytics import YOLO

# Load a segmentation model
model = YOLO("yolov8n-seg.pt")  
#也可以下载好模型，然后把模型路径塞进去，model=YOLO("D:/Python/Great Project/YOLO_Bicycle_Theft_Detection/src/yolov8n-seg.pt")
# 注意路径名若用反斜杠要加转义符：\ x \\ √
results=model(source="../testcases/test_mid_autumn_party.jpg", show=True,save=True)
```
其支持的输入类型很多，详情可查看：https://docs.ultralytics.com/modes/predict/#inference-sources

## 使用模式

1. #### Train

```python
from ultralytics import YOLO

# Load a model
#使用了 yolov8n.yaml 文件中定义的网络结构和超参数。这适用于您想要创建自己的 YOLO 模型或尝试使用不同的网络架构进行目标检测任务的情况。
model = YOLO('yolov8n.yaml')  # build a new model from YAML
#加载一个预训练好的 YOLO 模型，在训练时推荐使用。此时，模型的权重被初始化为预训练权重，可以快速地进行 Fine-tune 以适应不同的目标检测任务。
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
#结合了前两个步骤。它首先基于 yolov8n.yaml 构建了一个新的 YOLO 模型，然后从预训练的权重文件 yolov8n.pt 中加载权重进行 Fine-tune。这种方法可以让您同时使用自定义的网络结构和预训练的权重进行训练
model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='coco128.yaml', epochs=100, imgsz=640)

```

2. #### Val

   ```python
   from ultralytics import YOLO
   
   # Load a model
   model = YOLO('yolov8n.pt')  # load an official model
   model = YOLO('path/to/best.pt')  # load a custom model
   
   # Validate the model
   metrics = model.val()  # no arguments needed, dataset and settings remembered
   metrics.box.map    # map50-95
   metrics.box.map50  # map50
   metrics.box.map75  # map75
   metrics.box.maps   # a list contains map50-95 of each category
   
   ```

   

3. #### Predict

```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('path/to/best.pt')  # load a custom model

# Predict with the model
results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image

```



4. #### Track

   Ultralytics YOLO支持以下跟踪算法。可以通过传递相关的 YAML 配置文件来启用它们，例如：`tracker=tracker_type.yaml`

   - [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - 用于启用此跟踪器。`botsort.yaml`
   - [ByTeSORT](https://github.com/ifzhang/ByteTrack) - 用于启用此跟踪器。`bytetrack.yaml`

   默认跟踪器是BoT-SORT。

```python
from ultralytics import YOLO

# Load an official or custom model
model = YOLO('yolov8n.pt')  # Load an official Detect model
model = YOLO('yolov8n-seg.pt')  # Load an official Segment model
model = YOLO('yolov8n-pose.pt')  # Load an official Pose model
model = YOLO('path/to/best.pt')  # Load a custom trained model

# Perform tracking with the model
results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)  # Tracking with default tracker
results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # Tracking with ByteTrack tracker

```

mdel.track一些可能有用的参数，下面为示例

```python
from ultralytics import YOLO

# Configure the tracking parameters and run the tracker
model = YOLO('yolov8n.pt')
results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.3, iou=0.5, show=True)

```

一个视频多帧追踪的例子

```python
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "path/to/video.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

```

## 处理YOLO输出结果
### 获得结果各部分
```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(['im1.jpg', 'im2.jpg'], stream=True)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
```
可以看到，模型输出结果返回Results对象的列表，Results对象包含了boxes、masks、keypoints、probs四个对象（当然还有其它属性），分别对应了检测框、分割区域、所处位置、置信度。

### 结果处理
可直接参考：https://docs.ultralytics.com/modes/predict/#working-with-results
```python

```






## YOLO训练数据集汇总
- https://bair.berkeley.edu/blog/2018/05/30/bdd/
- https://www.kaggle.com/datasets/dataclusterlabs/bicycle-image-dataset-vehicle-dataset