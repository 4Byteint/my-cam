# sub.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'test_topic',
            self.listener_callback,
            10
        )
        self.get_logger().info('✅ Subscriber node 啟動成功，等待接收訊息...')

    def listener_callback(self, msg):
        print("🔥 收到了！", msg.data)
        # self.get_logger().info(f'📥 收到訊息：{msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = MinimalSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
