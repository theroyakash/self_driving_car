# Reinforcement Learning for Self Driving Cars

This is an example of Self Driving car Using Carla and Reinforcement learning

## Introduction:
CARLA has been developed from the ground up to support development, training, and validation of autonomous driving systems. In addition to open-source code and protocols, CARLA provides open digital assets (urban layouts, buildings, vehicles) that were created for this purpose and can be used freely. The simulation platform supports flexible specification of sensor suites, environmental conditions, full control of all static and dynamic actors, maps generation and much more.

## Highlighted features
- Scalability via a server multi-client architecture: multiple clients in the same or in different nodes can control different actors.
- Flexible API: CARLA exposes a powerful API that allows users to control all aspects related to the simulation, including traffic generation, pedestrian behaviors, weathers, sensors, and much more.
- Autonomous Driving sensor suite: users can configure diverse sensor suites including LIDARs, multiple cameras, depth sensors and GPS among others.
- Fast simulation for planning and control: this mode disables rendering to offer a fast execution of traffic simulation and road behaviors for which graphics are not required.
- Maps generation: users can easily create their own maps following the OpenDrive standard via tools like RoadRunner.
- Traffic scenarios simulation: our engine ScenarioRunner allows users to define and execute different traffic situations based on modular behaviors.
- ROS integration: CARLA is provided with integration with ROS via our ROS-bridge
- Autonomous Driving baselines: Carla provide Autonomous Driving baselines as runnable agents in CARLA, including an AutoWare agent and a Conditional Imitation Learning agent.

## Reinforcement Learning:
CARLA consists mainly of two modules, the CARLA Simulator and the CARLA Python API module. The simulator does most of the heavy work, controls the logic, physics, and rendering of all the actors and sensors in the scene; it requires a machine with a dedicated GPU to run. The CARLA Python API is a module that you can import into your Python scripts, it provides an interface for controlling the simulator and retrieving data. With this Python API you can, for instance, control any vehicle in the simulation, attach sensors to it, and read back the data these sensors generate. Most of the aspects of the simulation are accessible from Carla's Python API, and more will be in future releases.

![screenshot](./Self%20Driving%20Car/sources/scripts%20image.png)

Sensors are a special type of actor able to measure and stream data. All the sensors have a listen method that registers the callback function that will be called each time the sensor produces a new measurement. Sensors are typically attached to vehicles and produce data either each simulation update, or when a certain event is registered.

To Learn more about the sensors read the carla documentation [here](https://carla.readthedocs.io/en/latest/cameras_and_sensors/)

Using sensor data and Reinforcement learning the model is trained. To Start training the model dowload the **Model Traing.py** file and run. Result for running model 100 episodes is this

![screenshot](./Self%20Driving%20Car/sources/tensorboard.png)

Running it for a few hundred thousand episodes like 200,000 to 500,000 will enable us to see anything decent, provided your computer doesn't blows-up.
