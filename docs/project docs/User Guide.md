# 用户指南


## 客户端




## 服务器端

本项目需要在milvus服务与mysql服务开启的情况下运行

### 启动milvus服务
在docker开启的情况下，运行以下命令启动milvus服务
```bash
cd <project_path>/env/milvus
docker-compose up -d
```

若容器已构建，可直接运行`docker start [OPTIONS] CONTAINER [CONTAINER...]`命令启动milvus服务


### 启动mysql服务
在mysql官网下载安装MySQL 5.x版本

安装完毕后，新建database命名为Bike_Database，运行以下命令启动mysql服务

```bash
mysql -h localhost -u root -p Bike_Database
```




