# Control Duckietown by Using mmWave Gesture Recongition

## Overview

### Introduction

This project extends mmWave gesture recognition capabilities to control Duckietown robot movements. Using the AWR1642 mmWave radar data, it provides Setup, Record, Train, and Predict functionalities to identify specific gestures and translate them into corresponding Duckiebot control commands. Supported gestures include:

- Swipe Up
- Swipe Down
- Swipe Left
- Swipe Right
- Clockwise Rotation
- Counterclockwise Rotation

> [!NOTE]
> The `Record` feature allows you to capture gestures for training.

### ROS Environment

This project uses ROS Kinetic and establishes communication through rosbridge for command and message transmission.

### ROS Subscribers and Publishers

1. Publishers:
   - Responsible for sending (pushing) messages
   - Example in your code:

    ```python
    # This is a publisher that sends Joy messages
    self.publisher = roslibpy.Topic(
        self.client,
        '/ubuntu_desktop/joy',
        'sensor_msgs/Joy'
    )
    ```

2. Subscribers:
   - Responsible for receiving messages
   - Triggered when new messages arrive

Publisher -----> Topic -----> Subscriber
(Your Program)  (/joy)    (joy_mapper_node)

Actual workflow:

- Your program (Publisher) sends Joy messages
- `joy_mapper_node` (Subscriber) receives these messages
- `joy_mapper_node` converts them to vehicle commands and becomes a new Publisher
- Motor control node as Subscriber receives these commands

Like this:

```text
Your Program --Joy msgs--> joy_mapper_node --Vehicle cmds--> Motor Control Node ---> Actual Motors
(Publisher)               (Sub+Pub)                        (Subscriber)
```

### Joystick signal

The joystick signals can be divided into two types:

- joy_msg.axes
  - Four axes: `joy_msg.axes = [0, 0, 0, 0]`
  - `joy_msg.axes[1]`
    - `=  1.0` : Turn left in place
    - `= -1.0` : Turn right in place
    - `=  0.0` : Stop
  - `joy_msg.axes[3]`
    - `=  1.0` : Move backward
    - `= -1.0` : Move forward
    - `=  0.0` : Stop
- joy_msg.buttons
  - 15 buttons: `joy_msg.buttons = [0] * 15`
    - `joy_msg.buttons[1] = 1` : Enable turning left/right
    - `joy_msg.buttons[3] = 1` : Enable forward/backward movement

## Prerequisites

### Duckietown Setup

#### Using SSH or UART Connection to Duckiebot

You can connect to the Duckiebot using either SSH (`ssh ubuntu@192.168.xxx.xxx`) or UART connection.

#### Test joystick Function

You can input the following command to test if the joystick is working.

```sh
jstest /dev/input/js0
```

After inputting the command, you will see the following image. You can move the joystick and observe if the values change.

#### Quick Start Using Joystick

This is a quick test to check if the joystick is working. You can first set the script, and then you don't need to input it again (the file name is `quick_start_joystick.sh`).

```python=
#!/bin/bash

# Switch to the duckietown directory
cd ~/duckietown

# Load the environment settings
source environment.sh

# Start the joystick control
roslaunch duckietown joystick.launch veh:=ubuntu_desktop
```

After setting the script, you can open the joystick, and the left and right sides of the joystick will only respond to forward and backward movements, and the buttons will not function.
If the script is set up, you can run the following file directly.

```sh
./quick_start_joystick.sh
```

### Computer Setup

- Flash the official demo firmware onto the AWR1642 development board before starting.

    [Initial Setup for AWR1642BOOST](https://gist.github.com/zyx1121/0756055fa9138aec81617501e2e5f263)

- Get `poetry` to manage the dependencies.

    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```

- Change the host in `cli.py` to your Duckietown's IP address.

    ```python=
    self.duckie = DuckieController(host="192.168.xxx.xxx", port=9090)
    ```

## Getting Started

### Duckietown

Launch these commands simultaneously on your Duckietown:

```sh
# ROS core
roscore
```

```sh
# ROS communication bridge (between client and server)
roslaunch rosbridge_server rosbridge_websocket.launch
```

```sh
# Launch joystick control in Duckietown system (you can use the `quick_start_joystick.sh` to start the joystick)
cd ~/duckietown
source environment.sh
roslaunch duckietown joystick.launch veh:=ubuntu_desktop
```

```sh
# Monitor incoming signals (optional)
rostopic echo /ubuntu_desktop/joy
```

### Computer

- Clone the repository and cd to the project directory

    ```sh
    git clone https://github.com/zyx1121/mmwave-gesture-recognition && cd mmwave-gesture-recognition
    ```

- Install dependencies

    ```sh
    poetry install
    ```

- Launch the console to start

    ```sh
    poetry run mmwave-gesture-recognition
    ```

- Setup the AWR1642 development board

    ```sh
    CLI > cfg
    ```

- Train the model

    ```sh
    CLI > train LSTM
    ```

- Plot the radar data in real-time and press `q` to exit

    ```sh
    CLI > plot
    ```

- Start predicting gestures and press `ctrl+c` to exit

    ```sh
    CLI > predict
    ```

### Gesture and Duckiebot Control Mapping

| Gesture | Duckiebot Action |
|---------|------------------|
| Swipe Up | Move Forward |
| Swipe Down | Move Backward |
| Swipe Left | Turn Left |
| Swipe Right | Turn Right |
| Clockwise / Counterclockwise | Stop |

## Project Structure

- `mmwave-gesture-recognition/`
    - `configs/` - (Configuration files)
        - `profile.cfg`
    - `models/` - (Trained models)
        - `Conv2D.keras`
        - `LSTM.keras`
    - `records/` - (Gesture data records)
        - `[label]_[%m%d%H%M%S].npy`
        - `...`
    - `src/` - (Source code)
        - `console/` - (Console interface)
            - `cli.py` - (Command-line interface)
        - `mmwave/` - (mmWave radar data processing)
            - `radar.py` - (Radar data processing)
            - `uart.py` - (UART communication)
        - `utils/` - (Utility functions)
            - `logger.py` - (Logging configuration)
        - `main.py` - (Main entry point)
    - `pyproject.toml` - (Poetry configuration)
    - `README.md` - (Project documentation)

## Command Functions

- `cfg` : Transmits settings from profile.cfg to the device.

- `plot` : Plots the radar data in real-time.

- `record` `[gesture]` `[times]` : Records the gesture [gesture] data [times] times and saves it to records/[gesture]_[date].npy.

## References

1. [Duckietown Car Starter Kit](https://piepie.com.tw/21576/duckietown-car-starter-kit)
2. [mmWave Gesture Recognition Project](https://github.com/zyx1121/mmwave-gesture-recognition)

## License

This project is licensed under the MIT License, which permits commercial use, modification, distribution, and private use.
