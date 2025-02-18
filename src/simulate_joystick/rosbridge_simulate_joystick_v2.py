import threading
import time

import roslibpy


class DuckieController:
    def __init__(self, host="192.168.68.117", port=9090):
        """Initialize DuckieController

        Args:
            host (str): ROS bridge server address
            port (int): ROS bridge server port
        """
        self._setup_ros_connection(host, port)
        self.running = True
        self.current_direction = "none"
        self.current_throttle = 0.0
        self._start_send_thread()

    def _setup_ros_connection(self, host, port):
        """Set up ROS connection"""
        try:
            # Check if there's an active client
            if hasattr(self, "client"):
                if self.client.is_connected:
                    try:
                        self.publisher.unadvertise()
                    except:
                        pass
                    self.client.terminate()
                # Important: Delete old client instance
                delattr(self, "client")
                time.sleep(1)  # Wait for connection to fully close

            # Establish new connection
            self.client = roslibpy.Ros(host=host, port=port)
            self.client.run()

            # Wait to ensure connection is established
            timeout = 5  # 5 seconds timeout
            start_time = time.time()
            while not self.client.is_connected:
                if time.time() - start_time > timeout:
                    raise ConnectionError("Connection timeout")
                time.sleep(0.1)

            self.publisher = roslibpy.Topic(self.client, "/ubuntu_desktop/joy", "sensor_msgs/Joy")
            self.publisher.advertise()
        except Exception as e:
            if hasattr(self, "client"):
                try:
                    if hasattr(self, "publisher"):
                        self.publisher.unadvertise()
                    self.client.terminate()
                    # Important: Delete failed client instance
                    delattr(self, "client")
                except:
                    pass
            raise ConnectionError(f"ROS connection failed: {e}")

    def _start_send_thread(self):
        """Start sending thread"""
        self.send_thread = threading.Thread(target=self._send_loop)
        self.send_thread.daemon = True
        self.send_thread.start()

    def _send_loop(self):
        """Continuous control signal sending loop"""
        while self.running and self.client.is_connected:
            try:
                current_time = time.time()

                # Create control signal
                axes = [0.0] * 4
                buttons = [0] * 15

                # Set direction
                if self.current_direction == "left":
                    axes[1] = 1.0
                elif self.current_direction == "right":
                    axes[1] = -1.0

                # Set throttle
                axes[3] = self.current_throttle

                # Set required buttons
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
                print(f"Send error: {e}")

    def move(self, direction="none", throttle=0.0):
        """Control Duckie movement

        Args:
            direction (str): Direction control ("none", "left", "right")
            throttle (float): Speed control (-1.0 to 1.0)
        """
        self.current_direction = direction
        self.current_throttle = max(min(throttle, 1.0), -1.0)  # Limit between -1.0 and 1.0

    def forward(self, speed=1.0):
        """Move forward"""
        self.move(throttle=abs(speed))

    def backward(self, speed=1.0):
        """Move backward"""
        self.move(throttle=-abs(speed))

    def turn_left(self):
        """Turn left"""
        self.move(direction="left")

    def turn_right(self):
        """Turn right"""
        self.move(direction="right")

    def stop(self):
        """Stop movement"""
        self.move()

    def close(self):
        """Close controller and clean up resources"""
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
                # Important: Delete client instance
                delattr(self, "client")
                time.sleep(1)  # Wait for connection to fully close
            except:
                pass

    def __enter__(self):
        """Support for with statement"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Auto cleanup when with statement ends"""
        self.close()
