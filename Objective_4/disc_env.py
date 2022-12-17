# -*- coding: utf-8 -*-


import gym
import numpy as np
import matplotlib.pyplot as plt 
import os
import math
import gym
import matplotlib.pyplot as plt
import pygame
import pygame.camera
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt 
import random

from torch.distributions import Categorical
from gym import spaces
from itertools import count
from collections import namedtuple

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class DiffDriveEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=4):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window'

        self.max_v = 1
        self.max_w = 2
        self._time_elapsed = 0
        self.high = self.size/2        
        # self.observation_space = spaces.Box(low=-high, high=high)
        self.observation_space = spaces.Dict({
                "agent": spaces.Box(-self.high, self.high, shape=(3,), dtype=np.float32),
                "target": spaces.Box(-self.high, self.high, shape=(2,), dtype=np.float32),
                "iterations": spaces.Box(0, 10000, shape=(2,), dtype=int),
                "obstical": spaces.Box(-self.high, self.high, shape=(2,), dtype=np.float32)
            })
        
        self._obstical_location = np.array([self.high/2, self.high/2],dtype=np.float32)   

        
        self.action_space = spaces.Discrete(8)
        
        self._action_to_velocity = {
            0: np.array([self.max_v, 0]),
            1: np.array([-self.max_v, 0]),
            2: np.array([self.max_v, self.max_w]),
            3: np.array([self.max_v, -self.max_w]),
            4: np.array([-self.max_v, self.max_w]),
            5: np.array([-self.max_v, -self.max_w]),
            6: np.array([0, self.max_w]),
            7: np.array([0, -self.max_w]),
        }
        
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
    def _get_obs(self):
        # return np.concatenate([self._agent_location[[0,1]], self._target_location], axis=0, dtype=np.float32)
        return {"agent": self._agent_location,
                "target": self._target_location,  
                "obstical": self._obstical_location,
                "time": self._time_elapsed}

    def _get_info(self):
        # return np.linalg.norm(self._agent_location[[0,1]] - self._target_location, ord=1)
        return {"distance": np.linalg.norm(self._agent_location[[0,1]] - self._target_location, ord=1)}
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        high = self.size/2
        # Choose the agent's location uniformly at random
        # self._agent_location = np.array(self.np_random.uniform(low=-2*self.high/3, high=2*self.high/3,size=3),dtype=np.float32)
        self._agent_location = np.array([self.high/1.5, 0, 0],dtype=np.float32)   
        self._agent_location[2] = np.array(self.np_random.uniform(low=0, high=2*np.pi,size=1),dtype=np.float32)                          
        # self._target_location = self.np_random.uniform(low=0, high=0,size=2)
        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = np.array([-self.high/1.5, 0],dtype=np.float32)   

        # self._target_location = np.array(self.np_random.uniform(low=-2*self.high/3, high=2*self.high/3,size=2),dtype=np.float32)
        # while np.array_equal(self._target_location, self._agent_location[[0,1]]):
        #     self._target_location = self.np_random.uniform(low=-high, high=high,size=2)
        self._obstical_location = np.array([0, 0],dtype=np.float32)   

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def step(self, action):
        h = .1
        self.max_time = 75
        completed = False
        terminated = False
        reward = 0
        pix_square_size = self.window_size / self.size
        circle_size = pix_square_size / 8

        actionvw = self._action_to_velocity[action]
        
        distance_from_object = np.sqrt((self._agent_location[0]-self._target_location[0])**2
                             +(self._agent_location[1]-self._target_location[1])**2)
        
        desired_heading = math.atan2(self._target_location[1]-self._agent_location[1], 
                                   self._target_location[0]-self._agent_location[0])

        heading_error_old = abs(self.normalize_angle(desired_heading-self._agent_location[2]))
        if heading_error_old > np.pi/2:
            heading_error_old = abs(heading_error_old - np.pi)
            
        distance_from_object_old = np.sqrt((self._agent_location[0]-self._target_location[0])**2
                             +(self._agent_location[1]-self._target_location[1])**2)
    
        self._agent_location[0] = self._agent_location[0] + h*actionvw[0]*np.cos(self._agent_location[2])
        self._agent_location[1] = self._agent_location[1] + h*actionvw[0]*np.sin(self._agent_location[2])
        self._agent_location[2] = self._agent_location[2] + h*actionvw[1]
        
        self._agent_location[2] = self.normalize_angle(self._agent_location[2])

        distance_from_object = np.sqrt((self._agent_location[0]-self._target_location[0])**2
                             +(self._agent_location[1]-self._target_location[1])**2)
        
        desired_heading = math.atan2(self._target_location[1]-self._agent_location[1], 
                                   self._target_location[0]-self._agent_location[0])
        distance_from_obstical = np.sqrt((self._agent_location[0]-self._obstical_location[0])**2
                                         +(self._agent_location[1]-self._obstical_location[1])**2)
        heading_error = abs(self.normalize_angle(desired_heading-self._agent_location[2]))
        if heading_error > np.pi/2:
            heading_error = heading_error - np.pi

        if distance_from_object < distance_from_object_old:
            reward += 1
        else:
            reward += -.5
            
        reward += -heading_error**2
        
        # reward = -4*distance_from_object-heading_error**2
        if distance_from_obstical < circle_size/pix_square_size:  
            reward += -50
            terminated = True
            
        if distance_from_object < circle_size/pix_square_size*.7:  
            reward = 100
            terminated = True
            completed = True
            
        elif abs(self._agent_location[0]) > self.size/2 or abs(self._agent_location[1] > self.size/2):
            terminated = True
            reward += -50
            
        elif self._time_elapsed >= self.max_time:
            terminated = True
            reward += -50
    
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
            
        if terminated:
            self._time_elapsed = 0
        else:
            self._time_elapsed += 1
            
        return observation, reward, terminated, info, completed
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels
        
        # First we draw the target
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (np.array([self._target_location[0] + self.size/2,
                       -self._target_location[1] + self.size/2])) * pix_square_size,
            pix_square_size / 8,
        )
    
        circle_size = pix_square_size / 8
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (np.array([self._agent_location[0] + self.size/2,
                       -self._agent_location[1] + self.size/2])) * pix_square_size,
            circle_size,
        )
        # Draw heading 
        pygame.draw.circle(
            canvas,
            (0, 0, 0),
            (np.array([self._agent_location[0]+ (circle_size/pix_square_size*.9)*np.cos(self._agent_location[2]) + self.size/2,
                       -(self._agent_location[1]+ (circle_size/pix_square_size*.9)*np.sin(self._agent_location[2])) + self.size/2])) * pix_square_size,
            circle_size/3,
        )
        pygame.draw.circle(
            canvas,
            (0, 255, 0),
            (np.array([self._obstical_location[0] + self.size/2,
                       -self._obstical_location[1] + self.size/2])) * pix_square_size,
            circle_size*1.5,
        )
    
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def normalize_angle(self,theta):
        if theta > np.pi:
            theta = theta - 2*np.pi
        elif theta < - np.pi:
            theta = theta + 2*np.pi 
        return theta
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
