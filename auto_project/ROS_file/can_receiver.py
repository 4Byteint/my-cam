import can
from can_utils import CANInterface



def receive_can_message():
    can = CANInterface()
    print("等待接收 CAN 訊息...")

    while True:
        msg = can.receive(timeout=10.0)
        if msg:
            if msg.arbitration_id == 0x123:
                print("處理特定 ID 封包")
        else:
            print("目前沒收到資料")

if __name__ == "__main__":
    receive_can_message()
