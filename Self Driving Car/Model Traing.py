'''
Copyright (c) 2019 theroyakash

see privacy policy at www.iamroyakash.com/privacy

This work is licensed under the terms of the GNU General Public License v3.0.

Permissions of this strong copyleft license are conditioned on making available complete source code of
licensed works and modifications which include larger works using a licensed work, under the same license.
Copyright and license notices must be preserved. Contributors provide an express grant of patent rights.
'''

import carla
import glob
import os
import random
import sys
import time
import numpy as np
import cv2
import math
import tensorflow as tf
import keras.backend.tensorflow_backend as backend

from threading import Thread
from keras.layers import Dense, GlobalAveragePooling2D
from keras.application.xception import Xception
from keras.optimizers import Adam
from keras.models import Model
from tqdm import tqdm
from collections import deque
from tensorflow import keras
from keras.callbacks import TensorBoard

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

actor_list = []

IM_WIDTH = 640
IM_HEIGHT = 480

SHOW_PREVIEW = False
SECONDS_PER_EPISODE = 10
REPLAY_MEMORY_SIZE = 5000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 16
PREDICTIONBATCH_SIZE = 1
TRAININGBATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5
MODEL_NAME = "Xception"
MEMORY_FRACTION = 0.7
MIN_REWARD = -200

EPISODES = 100
DISCOUNT = 0.99
epsilon = 2
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001

AGGREGATE_STATS_EVERY = 10

# Modified Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)


class CarEnv:
	SHOW_CAM = SHOW_PREVIEW
	STEER_AMT = 1.0  # Steer 1 = right, 0 = Straight, -1 = left
	im_width = IM_WIDTH
	im_height = IM_HEIGHT
	front_camera = None

	def __init__(self):
		self.client = carla.Client("localhost", 2000)  # Creates a server at local SSD, here inside the car for faster processing.
		# If we do process this thing to a remote server you know what can happen (964 ms ping, slow/no internet etc to fu*k the 
		# autonomus navigation)
		self.client.set_timeout(3.0)
		self.world = client.get_world()
		self.blueprint_library = self.world.get_blueprint_library()
		self.model_3 = self.blueprint_library.filter("model3")[0]  # Model 3 Tesla

	def reset(self):
		self.collison_hist = []
		self.actor_list = []

		self.transform = random.choice(world.get_map().get_spawn_points())  # I've the damn car now
		self.vehicle = world.spawn_actor(self.model_3, self.transform)  # Spawning a Tesla Model 3

		self.actor_list.append(self.vehicle)
	

	# self.vehicle.set_autopilot(True)  

	# This is a hard-coded game engine based self driving car, not the self-driven Car. 
	# We can make a model learn this auto-vehicular interaction(rule-based, game-engine based behaviour, 
	# not human-centered) but here we wanna make rules on our own, not game-engine engineered weird piece of Shite.

		# Use of Camera 
		self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
		self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")  # Why? Have to feed it to a neural net
		self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
		self.rgb_cam.set_attribute("fov", f"110")  # Fish Eye View from the camera

		transform = carla.Transform(carla.Location(x=2.5, z=0.7))  # setting a place for the camera
		self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
		self.actor_list.append(self.sensor)

		# Listening Data from the sensor
		self.sensor.listen(lambda data: process_img(data))

		self.vehicle.apply_control(carla.VehicleControl(throttle = 0.0, brake=0.0))
		time.sleep(4)

		col_sensor = self.blueprint_library.find("sensor.other.collison")
		self.col_sensor = self.world.spawn_actor(col_sensor, transform, attach_to=self.vehicle)
		self.actor_list.append(col_sensor)

		# Listening Data from the sensor
		self.col_sensor.listen(lambda event: self.collison.data(event))

		while self.front_camera is None:
			time.sleep(0.01)

		self.episode_start = time.time()

		return self.front_camera

	def collison_data(self, event):
		self.collison_hist.append(event)


	def process_img(self, image):
		i = np.array(image.raw_data)
		# Shape of a photo:
		i2 = i.reshape((self.im_height, self.im_width), 4) # RGBA Red-Green-Blue-Alpha from camera
		i3 = i2[:, :, :3]  # [All height, All width data, Only RGB data, fu*k the alpha alright]
		# There is a slower method to fu*k the alpha value from RGBA in a OpenCV function
		if self.SHOW_CAM:
			cv2.imshow("", i3)
			cv2.waitKey(1)
		self.front_camera = i3

	def step(self, action):
		if action == 0:
			self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
		elif action == 1:
			self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
		elif action == 2:
			self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))

		v = self.vehicle.get_velocity()
		kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

		if len(self.collison_hist != 0):
			done = True
			reward = -200

		elif kmh < 50:
			done = False
			reward = -1

		else:
			done = False
			reward = 1

		if self.episode_start + SECONDS_PER_EPISODE < time.time():
			done = True

		return self.front_camera, reward, done, None

# Implementations for Re-enforcement Learning:

class DQNAgent:
	def __int__(self):
		self.model = self.create_model()
		self.target_model.set_weights(self.model.get_weights())

		self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
		self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
		
		self.target_update_counter = 0
		self.graph = tf.get_default_graph()

		self.terminate = False
		self.last_logged_episode = 0
		self.training_initialized = False

	def create_model(self):
		base_model = Xception(weights=None, include_top=False, input_shape=(IM_HEIGHT,IM_WIDTH,3))
		
		x = base_model.output
		x = GlobalAveragePooling2D()(x)

		predictions = Dense(3, activation = "linear")(x)
		model = Model(inputs = base_model.input, output = predictions)
		model.compile(loss="mse", optimizer=Adam(lr=0.001),metrics=["accuracy"])
		return model

	def update_replay_memory(self, transition):
		print("code is on its way")
		self.replay_memory.append(transition)

	def train(self):
		if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
			return

		minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
		current_states = np.array([transition[0] for transition in minibatch]) / 255
		with self.graph.as_default():
			current_qs_list = self.model.predict(current_states, PREDICTIONBATCH_SIZE)


		new_current_states = np.array([transition[3] for transition in minibatch]) / 255
		with self.graph.as_default():
			future_qs_list = self.target_model.predict(new_current_states, PREDICTIONBATCH_SIZE)

		X = []
		y = []

		for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
			if not done:
				max_future_q = np.max(future_qs_list[index])
				new_q = reward + DISCOUNT * max_future_q
			else:
				new_q = reward

			current_qs = current_qs_list[index]
			current_qs[action] = new_q

			X.append(current_state)
			y.append(current_qs)

		# Tensorboard
		log_step = False
		if self.tensorboard.step > self.last_logged_episode:
			log_step = True
			self.last_logged_episode = self.tensorboard.step

		with self.graph.as_default():
			self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAININGBATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_step else None)
		
		if log_step:
			self.target_update_counter += 1

		if self.target_update_counter > UPDATE_TARGET_EVERY:
			self.target_update_model.set_weights(self.model.get_weights())
			self.target_update_counter = 0


	def get_qs(self, state):
		return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

	def train_in_loop(self):
		X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3).astype(np.float32))
		y = np.random.uniform(size=(1, 3).astype(np.float32))

		with self.graph.as_default():
			self.model.fit(X,y, verbose=False, batch_size=1)

		self.training_initialized = True

		while True:
			if self.terminate:
				return
			self.train()
			time.sleep(0.01)

if __name__ == "__main__":
	FPS = 20
	ep_rewards - [-200]

	random.seed(1)
	np.random.seed(1)
	tf.set_random_seed(1)

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=MEMORY_FRACTION)
	backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))

	if not os.path.isdir("models"):
		os.makedirs("models")

	agent = DQNAgent()
	env= CarEnv()

	trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
	trainer_thread.start()

	while not agent.training_initialized:
		time.sleep(0.01)

	agent.get_qs(np.ones((env.im_height, env.im_width, 3)))

	for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="episodes"):
		env.collison_hist = []
		agent.tensorboard.step = episode
		episode_reward = 0
		step = 1
		current_state = env.reset()
		done = False
		episode_start = time.time()

		while True:
			if np.random.random() > epsilon:
				action = np.argmax(agent.get_qs(current_state))
			else:
				action = np.random.randint(0,3)
				time.sleep(1/FPS)

			new_state, reward, done, _ = env.step(action)
			episode_reward += reward

			agent.update_replay_memory((current_state, action, reward, new_state, done))
			step += 1

			if done:
				break

		for actor in env.actor_list:
			actor.destroy()
            
        # Append episode reward to a list and log stats (every given number of episodes)
        
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if min_reward >= MIN_REWARD:
                agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
            
    # Set termination flag for training thread and wait for it to finish
    
    agent.terminate = True
    trainer_thread.join()
    agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
