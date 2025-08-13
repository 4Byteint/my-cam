import time
import os

log_file = "temperature_ros_cam.txt"

def get_cpu_temp():
    temp_str = os.popen("vcgencmd measure_temp").readline()
    return temp_str.strip()

with open(log_file, "a") as f:
    while True:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        temp = get_cpu_temp()
        log_line = f"[{timestamp}] CPU 溫度：{temp}"
        print(log_line)
        f.write(log_line + "\n")
        f.flush()
        time.sleep(2)
