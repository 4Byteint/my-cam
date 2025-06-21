# socket_sender.py
import socket
import json
import time

HOST = '127.0.0.1'  # 或 ROS Node 的 IP
PORT = 5005

def send_data(x, y, angle):
    data = json.dumps({"x": x, "y": y, "angle": angle})
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(data.encode('utf-8'))
        
        


# 測試用
if __name__ == "__main__":
    while True:
        send_data(120, 200, 45.2)
        time.sleep(1)
