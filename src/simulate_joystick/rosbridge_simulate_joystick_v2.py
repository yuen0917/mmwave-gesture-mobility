import threading
import time

import roslibpy


class DuckieController:
    def __init__(self, host="192.168.68.117", port=9090):
        """初始化 DuckieController

        Args:
            host (str): ROS bridge 伺服器位址
            port (int): ROS bridge 伺服器埠號
        """
        self._setup_ros_connection(host, port)
        self.running = True
        self.current_direction = "none"
        self.current_throttle = 0.0
        self._start_send_thread()

    def _setup_ros_connection(self, host, port):
        """設置 ROS 連接"""
        try:
            # 檢查是否已經有活躍的 client
            if hasattr(self, "client"):
                if self.client.is_connected:
                    try:
                        self.publisher.unadvertise()
                    except:
                        pass
                    self.client.terminate()
                # 重要：刪除舊的 client 實例
                delattr(self, "client")
                time.sleep(1)  # 等待連接完全關閉

            # 建立新的連接
            self.client = roslibpy.Ros(host=host, port=port)
            self.client.run()

            # 等待確保連接建立
            timeout = 5  # 5秒超時
            start_time = time.time()
            while not self.client.is_connected:
                if time.time() - start_time > timeout:
                    raise ConnectionError("連接超時")
                time.sleep(0.1)

            self.publisher = roslibpy.Topic(self.client, "/ubuntu_desktop/joy", "sensor_msgs/Joy")
            self.publisher.advertise()
        except Exception as e:
            if hasattr(self, "client"):
                try:
                    if hasattr(self, "publisher"):
                        self.publisher.unadvertise()
                    self.client.terminate()
                    # 重要：刪除失敗的 client 實例
                    delattr(self, "client")
                except:
                    pass
            raise ConnectionError(f"ROS 連接失敗：{e}")

    def _start_send_thread(self):
        """啟動發送執行緒"""
        self.send_thread = threading.Thread(target=self._send_loop)
        self.send_thread.daemon = True
        self.send_thread.start()

    def _send_loop(self):
        """持續發送控制訊號的迴圈"""
        while self.running and self.client.is_connected:
            try:
                current_time = time.time()

                # 建立控制訊號
                axes = [0.0] * 4
                buttons = [0] * 15

                # 設定方向
                if self.current_direction == "left":
                    axes[1] = 1.0
                elif self.current_direction == "right":
                    axes[1] = -1.0

                # 設定油門
                axes[3] = self.current_throttle

                # 設定必要按鈕
                buttons[1] = 1
                buttons[3] = 1

                joy_msg = {
                    "axes": axes,
                    "buttons": buttons,
                    "header": {"stamp": {"secs": int(current_time), "nsecs": int((current_time % 1) * 1e9)}, "frame_id": ""},
                }

                self.publisher.publish(roslibpy.Message(joy_msg))
                time.sleep(0.02)  # 50Hz

            except Exception as e:
                print(f"發送錯誤: {e}")

    def move(self, direction="none", throttle=0.0):
        """控制小鴨移動

        Args:
            direction (str): 方向控制 ("none", "left", "right")
            throttle (float): 速度控制 (-1.0 到 1.0)
        """
        self.current_direction = direction
        self.current_throttle = max(min(throttle, 1.0), -1.0)  # 限制在 -1.0 到 1.0 之間

    def forward(self, speed=1.0):
        """向前移動"""
        self.move(throttle=abs(speed))

    def backward(self, speed=1.0):
        """向後移動"""
        self.move(throttle=-abs(speed))

    def turn_left(self):
        """向左轉"""
        self.move(direction="left")

    def turn_right(self):
        """向右轉"""
        self.move(direction="right")

    def stop(self):
        """停止移動"""
        self.move()

    def close(self):
        """關閉控制器並清理資源"""
        self.running = False
        if hasattr(self, "send_thread"):
            self.send_thread.join(timeout=1.0)
        if hasattr(self, "publisher"):
            try:
                self.publisher.unadvertise()
            except:
                pass
        if hasattr(self, "client"):
            try:
                self.client.terminate()
                # 重要：刪除 client 實例
                delattr(self, "client")
                time.sleep(1)  # 等待連接完全關閉
            except:
                pass

    def __enter__(self):
        """支援 with 語句"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """支援 with 語句結束時自動清理"""
        self.close()
