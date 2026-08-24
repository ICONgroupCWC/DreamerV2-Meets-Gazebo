# DreamerV2 Meets Gazebo — C-JEPA & W-JEPA

This repository contains the implementation of **Control-JEPA (C-JEPA)** and **Wireless-JEPA (W-JEPA)** for predictive and communication-aware robotic control.

## Dependencies

- Python
- PyTorch
- Gym / Gymnasium
- ROS Noetic
- Gazebo Classic
- OpenCV
- NumPy

## Base Implementation

The control world model is built on the original **DreamerV2** implementation:

https://github.com/danijar/dreamerv2

## Training

The model is first trained in a **Gym-based racing-car environment**.

```text
Gym Racing Environment
        ↓
DreamerV2 / C-JEPA Training
        ↓
Saved Model Weights
