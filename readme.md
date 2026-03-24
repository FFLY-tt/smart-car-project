# Vehicle Autonomous Driving Obstacle Avoidance Model Based on GAP SAC and A Star Algorithms

This repository contains the code and documentation for our autonomous driving obstacle avoidance system developed for Challenge 2 DEEP Reinforcement Learning for Obstacle Avoidance. The project enables a miniature intelligent car to automatically avoid dynamic and static obstacles in complex maps using Gazebo and ROS2.

## Key Innovations and Features
* Hybrid State Machine Architecture: Breaks down complex scenes into local ones and smoothly switches control between the A star algorithm and the GAP SAC trained model with zero delay.
* 2D LiDAR to BEV Conversion: Reconstructs 360 discrete 1D LiDAR ranging points into a 64 by 64 resolution 2D bird eye view in real time to improve information perception.
* Multi Frame Stacking: Continuously stacks four consecutive image frames in the background to accurately identify dynamic objects and prevent model memory gaps during control transfers.
* Physical Boundary Detection: Distinguishes corners and obstacles by setting a continuous obstacle length of 0.8 meters and utilizes a 0.22 meter collision distance to effectively identify collisions.
* Posture Stabilization and Anti Backtracking: Introduces a posture stabilizer to realign the vehicle to the lane center after obstacle avoidance and uses dynamic A star pathpoint updates to prevent the vehicle from backtracking.

## System Architecture
The system adopts a layered and decoupled hybrid control architecture divided into three main dimensions including the perception layer, the decision making layer, and the execution layer. 
* Perception and Data Conversion Module: Converts 2D LiDAR data into a 4 channel 64 by 64 2D matrix.
* Global Path Planning Module: Uses the A star algorithm with Manhattan distance to generate the optimal path and includes a pure tracker module.
* Local Obstacle Avoidance Module: Activated when threats are detected within 3.0 meters ahead, taking full control of speed and steering using GAP SAC.
* Central Arbitration and Stabilization Module: A finite state machine that handles module transitions and vehicle stabilization.

## Platforms and Technologies
* Software: ROS2 Humble for millisecond level low latency communication.
* Simulation: Gazebo for high fidelity chassis dynamics gravity and friction.
* Deep Learning: PyTorch framework.
* Hardware Target: Standardized miniature intelligent car chassis equipped with a 2D single line LiDAR.

## Performance and Outcomes
Tested in a Gazebo miniature city intersection scenario containing static bollards and dynamic pedestrians.
* Pure Navigation Baseline: Approximately 50 seconds to complete the global path.
* Mixed Interference Test: Approximately 55 seconds to complete the path with multiple SAC obstacle avoidance maneuvers and state machine handovers.
* Result: Achieved zero collision closed loop navigation with extremely high navigation consistency and no meaningless stationary maneuvers or detours.

## Team
Group 8 Ontario Tech University
* Fangyu Lin
* Hiranmayee Brundavanam

Repository Link: https://github.com/FFLY-tt/smart-car-project