import rclpy
from rclpy.node import Node
import subprocess

class TimerExecutor(Node):
    def __init__(self):
        super().__init__('timer_executor')
        self.timer_period = 5  # 秒
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.program_index = 0
        self.programs = [
            ['/usr/bin/python3', '/app/my_project/first_model_zoo.py'],
            ['/usr/bin/python3', '/app/my_project/yolo11.py']
        ]

    def timer_callback(self):
        current_program = self.programs[self.program_index]
        
        # 终止旧进程
        if hasattr(self, 'process'):
            if self.process.poll() is None:
                self.get_logger().info("Terminating previous process...")
                self.process.terminate()
                try:
                    self.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()

        # 启动新进程
        self.process = subprocess.Popen(
            current_program,
            stdout=open('/tmp/script_output.log', 'a'),
            stderr=subprocess.STDOUT
        )
        self.get_logger().info(f'Running: {current_program}')

        # 切换索引
        self.program_index = (self.program_index + 1) % len(self.programs)

def main(args=None):
    rclpy.init()
    node = TimerExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'process') and node.process.poll() is None:
            node.process.terminate()
            node.process.kill()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()