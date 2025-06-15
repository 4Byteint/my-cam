FROM ros:humble-ros-base

# 設定環境參數
ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# 工作目錄
WORKDIR /workspace

# 複製你的 pub/sub 程式進來
COPY pub.py .
COPY sub.py .

# 安裝 Python 相依套件
RUN apt update && apt install -y python3-pip 

CMD ["bash"]
