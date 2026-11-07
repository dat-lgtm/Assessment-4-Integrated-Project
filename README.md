# Assessment-4-Integrated-Project
# UTS 41014 Track A: TurtleBot3 navigation and docking

This project uses MuJoCo to simulate a TurtleBot3 for UTS 41014 Sensors and Control in Mechatronics Systems. The aim is to drive around obstacles and dock in front of a charging station using an ArUco marker.

The robot starts in the bottom-left corner of a 5 m × 5 m room, at (-2, -2), facing the centre. The charging station is in the top-right corner at (2, 2). 
## How it works

- Wheel encoders measure wheel movement. The Extended Kalman Filter (EKF) uses these readings to predict the robot position and heading.
- The 2D LiDAR detects obstacles. Matching its readings to the known map also helps estimate where the robot is.
- Near the station, the camera detects an ArUco marker and measures the robot position and heading relative to it.
- The EKF corrects its prediction when camera or LiDAR measurements are available.
- A* finds a route around the obstacles. The LQR controller follows that route using the EKF estimate.
- During docking, a separate feedback controller brings the robot to a stop in front of the station.

Before navigation starts, the robot runs two short calibration tests to compare wheel encoder movement with LiDAR measurements. It then uses the resulting scale factors during navigation. The program starts with a known map, marker position and robot starting pose.

## Required files

Keep the following files in the project folder. Leave the files inside `assets/` in their original locations:

```text
UTS-41014-Track-A/
├── main.py
├── simulation.py
├── robot.py
├── robot_model.py
├── sensors.py
├── noise.py
├── ekf.py
├── navigation.py
├── controller.py
├── evaluation.py
├── compatible_viewer.py
├── scene.xml
├── assets/
├── results/
├── requirements.txt
└── README.md
```

The `assets/` folder contains the robot meshes, marker image and ROBOTIS licence. 

## Setup

This project was tested on Windows with Python 3.11. Download the repository, extract it, and open the project folder in Visual Studio Code.

In the VS Code integrated terminal, create a virtual environment and install the packages:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Press **Ctrl+Shift+P**, choose **Python: Select Interpreter**, and select `.venv\Scripts\python.exe`.

## Run the simulation

Once setup is complete, open `main.py` and click **Run Python File** in VS Code. 

You should see the 3D simulation, a camera view and a navigation map. The robot runs until it finishes docking or reaches the time limit. On the navigation map, the green line is the planned route and the robot marker shows the EKF position estimate.

3D view controls:

- Left-drag: rotate.
- Right-drag: pan.
- Mouse wheel: zoom.
- Double-click: reset the view.
- Press **Q** in a simulation window to stop.

The project displays the MuJoCo view through OpenCV. 


## Result graphs

Three figures appear after the run:

| Figure | What it shows |
|---|---|
| Navigation trajectory | Planned route, actual path, EKF estimate, obstacles and station |
| EKF consistency | Position and heading estimation errors, uncertainty bounds and NEES |
| Tracking and docking | Cross-track error, distance to the docking goal and heading error |


The terminal reports whether docking succeeded, whether the robot touched an obstacle, and how well the EKF estimated its position and heading. It also prints consistency metrics. Changing the noise or trial settings can change the results.

## Ground truth

Ground truth means the actual robot position and heading in MuJoCo. The function `truth_for_evaluation()` in `robot.py` reads it, and `simulation.py` records it to calculate errors. The controller and planner use the EKF estimate. The estimator and odometry calibration use sensor readings, without reading ground truth.

On the navigation trajectory graph, the black line is the actual path and the red line is the EKF estimate.

## File guide

| File | Purpose |
|---|---|
| main.py | Starts the program |
| simulation.py | Connects the simulation, sensors, estimator, planner and controller |
| robot.py | Loads the scene and provides wheel commands and simulated sensors |
| robot_model.py | Motion models, Jacobians and encoder conversion |
| sensors.py | Marker detection, LiDAR localisation and odometry calibration |
| noise.py | Noise and covariance settings |
| ekf.py | EKF prediction and measurement updates |
| navigation.py | Map and A* path planning |
| controller.py | LQR tracking, docking feedback and obstacle safety checks |
| evaluation.py | Performance metrics and result graphs |
| compatible_viewer.py | Interactive 3D display |
| scene.xml | Robot, room, obstacles and charging station |

## Limitations

The robot needs a known map, and its simulated dynamics are approximate. The camera uses a pinhole model with no lens distortion. The optional calibration check uses synthetic points, so it does not replace calibration using captured images. More trials are needed to check performance under different conditions. The simulation has not been validated for use on a physical robot.

## Tutorial acknowledgement and AI use

The motion model, LQR, EKF and consistency plots follow ideas from the Week 5 to 8 tutorials and Assessment 3. This project also includes camera marker detection, LiDAR map matching, path planning and docking.

ChatGPT/Codex helped adapt the tutorial code and set up MuJoCo. It also helped debug problems, create the result plots and edit this README.

## Asset attribution

TurtleBot3 visual meshes were adapted from the [ROBOTIS TurtleBot3 repository](https://github.com/ROBOTIS-GIT/turtlebot3). The original source URDF and `ROBOTIS_LICENSE` are in `assets/`. Include the licence when sharing the project.
## Individual Contribution Statement

**Contributor:** Tran Tuan Dat Nguyen - 14226258

**Project Type:** Individual Project 

This project was completed independently. I was responsible for the design, implementation, integration, testing, and documentation of the project.

