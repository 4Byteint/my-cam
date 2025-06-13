# can_interface.py
import can

class CANInterface:
    def __init__(self, channel='vcan0', bitrate=500000):
        self.bus = can.interface.Bus(channel=channel, bustype='socketcan')
        print(f"[CAN] Initialized on {channel} with {bitrate} bps")

    def send(self, can_id, data):
        """傳送 CAN 資料"""
        msg = can.Message(arbitration_id=can_id, data=data, is_extended_id=False)
        try:
            self.bus.send(msg)
            print(f"[CAN] 發送:ID={hex(can_id)} Data={data}")
        except can.CanError as e:
            print(f"[CAN] 傳送失敗：{e}")

    def receive(self, timeout=1.0):
        """接收 CAN 訊息(有 timeout)"""
        try:
            msg = self.bus.recv(timeout)
            if msg:
                print(f"[CAN] 接收:ID={hex(msg.arbitration_id)} Data={list(msg.data)}")
                return msg
        except can.CanError as e:
            print(f"[CAN] 接收失敗：{e}")
        return None

def main():
    can = CANInterface()

    # 發送一筆資料
    can.send(0x01, [0x10, 0x20, 0x30])

    # 等待接收回應
    response = can.receive()
    if response:
        print("回應資料：", response.data)
        
if __name__ == "__main__":
    main()
    