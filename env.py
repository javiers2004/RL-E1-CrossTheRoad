import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pygame
import os


class CrossTheRoadVisionEnv(gym.Env):
    """
    Grid values:
      0 = empty
      1 = car
      2 = agent
      3 = goal row
      6 = car trail (danger)
      7 = river (lethal unless on log)
      8 = log (safe)
    """
    metadata = {"render_modes": ["human"], "render_fps": 3}


    # To initialize the environment with custom parameters
    def __init__(self, height=14, width=12, vision=3,
                 car_spawn_prob=0.2, max_cars_per_lane=2,
                 trail_prob=0.2, seed=None):
        super().__init__()
        assert vision % 2 == 1 and vision >= 3  # Size of the observation window (must be odd and at least 3)
        self.height = height    # Number of rows
        self.width = width    # Number of columns
        self.vision = vision    # Size of the observation window
        self.car_spawn_prob = car_spawn_prob    # Probability of spawning a car in each lane each step
        self.max_cars_per_lane = max_cars_per_lane    # Maximum number of cars allowed in each lane
        self.trail_prob = trail_prob    # Probability of spawning a trail behind a car
        self.cell_size = 48  # Size of each cell in pixels for rendering
        self.action_space = spaces.Discrete(4)  # 0=up,1=down,2=left,3=right
        self.num_car_lanes = self.height - 3  # Exclude goal and 2 river rows
        self.observation_space = spaces.Box(low=0, high=8,  # Observation space (vision + log positions + traffic light states)
                                            shape=(vision * vision + 4 + 1,), 
                                            dtype=np.int8)

        # Initialize variables
        self.window = None
        self.clock = None
        self.random = random.Random(seed)
        self.np_random = np.random.RandomState(seed)
        self.lanes_dir = [self.random.choice([-1, 1]) for _ in range(1, self.height - 1)]  # Directions of car lanes
        base_path = os.path.dirname(os.path.abspath(__file__))

        # --- Load images with error handling ---
        def load_img(path):
            try:
                return pygame.image.load(path)
            except pygame.error:
                print(f"Warning: Image not found at {path}. Using placeholder.")
                return pygame.Surface((48, 48), pygame.SRCALPHA)

        self.raw_car_images = []    # Load car images
        for i in range(1, 6):
            img_path = os.path.join(base_path, "images", f"car{i}.png")
            self.raw_car_images.append(load_img(img_path))

        self.raw_smoke_images = []   # Load smoke images
        for i in range(1, 5):
            img_path = os.path.join(base_path, "images", f"smoke{i}.png")
            self.raw_smoke_images.append(load_img(img_path))

        self.raw_wood_image = load_img(os.path.join(base_path, "images", "log.png"))    # Load log image
        self.raw_water_image = load_img(os.path.join(base_path, "images", "water.png")) # Load water image

        # Initialize final images
        self.car_images = None
        self.smoke_images = None
        self.wood_image = None
        self.water_image = None

        self.reset()

#  Reset the environment to the initial state
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.random.seed(seed)
            self.np_random.seed(seed)

        self.agent_pos = [self.height - 1, self.width // 2] # Start at bottom center
        self.cars = []  # Empty list of cars at start
        self.trails = []  # Empty list of trails at start
        self.t = 0  # Time step counter
        self.crossed_river = False  # Whether the agent has crossed the river

        # Create the river rows and logs
        self.river_row1 = self.random.randint(1, self.height - 3)
        self.river_row2 = self.river_row1 + 1
        log_start1 = self.random.randint(0, self.width - 3)
        log_start2 = self.random.randint(0, self.width - 3)
        self.logs_row1 = list(range(log_start1, log_start1 + 3))
        self.logs_row2 = list(range(log_start2, log_start2 + 3))

        #   Initialize traffic lights (1=green, 0=red) for each car lane
        self.traffic_lights = [1 for _ in range(1, self.height - 1)]
        self.traffic_light_timer = 0
        self.traffic_light_interval = 5

        return self._get_obs(), {}


# Step the environment
    def step(self, action):
        self.t += 1 # Increment time step
        reward = 0.0    # Initialize reward
        terminated = False  # Whether the episode is terminated
        truncated = False   # Whether the episode is truncated

        # Update traffic lights
        self.traffic_light_timer += 1
        if self.traffic_light_timer >= self.traffic_light_interval:
            self.traffic_light_timer = 0
            # Only car lanes (not river rows) switch lights
            for i in range(len(self.traffic_lights)):
                lane_row = i + 1
                if lane_row not in [self.river_row1, self.river_row2]:
                    self.traffic_lights[i] = 1 - self.traffic_lights[i]

        # Save agent's previous position
        old_row = self.agent_pos[0]
        
        # Agent movement 
        if action == 0 and self.agent_pos[0] > 0:  # UP direction
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.height - 1:  # DOWN direction
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:  # LEFT direction
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.width - 1:  # RIGHT direction
            self.agent_pos[1] += 1

        #Movement reward/penalty
        # ---------------------------------
        if self.agent_pos[0] < old_row:
            reward += 0.9  # Move forward (+1-0.1 reward)
        elif self.agent_pos[0] > old_row:
            reward -= 1.1  # Move backward (-1-0.1 penalty)
        else:
            reward -= 0.1  # Wait/Horizontal movement (-0.1 penalty)
        # ---------------------------------

        # Move logs
        self.logs_row1 = [(c + 1) % self.width for c in self.logs_row1] # Move right
        self.logs_row2 = [(c - 1) % self.width for c in self.logs_row2] # Move left

        # Move cars (only if traffic light is green)
        new_trails = []
        for car in self.cars:
            row_idx = car['row'] - 1
            is_green = self.traffic_lights[row_idx] == 1    
            if is_green:
                dir = self.lanes_dir[row_idx]
                new_positions = []
                for p in car['positions']:
                    next_p = p + dir

                    # Check for collision with moving car
                    if car['row'] == self.agent_pos[0]:
                        if next_p == self.agent_pos[1] or (p == self.agent_pos[1] and next_p == self.agent_pos[1]):
                            reward = -150.0  # (-150 penalty)
                            terminated = True
                    # Create trail
                    if car.get('trail', False):
                        new_trails.append({'row': car['row'], 'col': p, 'ttl': 5})
                    if 0 <= next_p < self.width:
                        new_positions.append(next_p)
                car['positions'] = new_positions
            else:
                # Check for collision with stationary car
                for p in car['positions']:
                    if car['row'] == self.agent_pos[0] and p == self.agent_pos[1]:
                        reward = -150.0  # (-150 penalty)
                        terminated = True
                        break
                    
        # Remove cars that have moved out of bounds
        self.trails.extend(new_trails)
        self.cars = [car for car in self.cars if car['positions']]

        # Spawn new cars (same logic as before)
        for r in range(1, self.height - 1):
            if r in [self.river_row1, self.river_row2]:
                continue
            dir = self.lanes_dir[r - 1]
            cars_in_row = [car for car in self.cars if car['row'] == r]
            n_existing = len(cars_in_row)
            # Spawn new car if below max and random chance is True
            if n_existing < self.max_cars_per_lane and self.random.random() < self.car_spawn_prob:
                spawn_col = 0 if dir == 1 else self.width - 1
                occupied = {p for car in cars_in_row for p in car['positions']}
                # If the spawn position is not occupied, spawn the car
                if spawn_col not in occupied:
                    self.cars.append({
                        'row': r,
                        'positions': [spawn_col],
                        'trail': self.random.random() < self.trail_prob,    # Probability of leaving a trail
                        'img_index': self.random.randint(0, len(self.raw_car_images) - 1)
                    })

        # Update trails 
        for tr in self.trails:
            tr['ttl'] -= 1
        self.trails = [t for t in self.trails if t['ttl'] > 0]

        # Check collisions with trails
        if not terminated:
            for tr in self.trails:
                if tr["row"] == self.agent_pos[0] and tr["col"] == self.agent_pos[1]:
                    reward = -150.0  # (-150 penalty)
                    terminated = True
                    break

        # Check river and logs
        if not terminated:
            if self.agent_pos[0] == self.river_row1:
                # Falling in the first river row
                if self.agent_pos[1] not in self.logs_row1:
                    reward = -150.0  # (-150 penalty)
                    terminated = True
                # Crossing the first river row
                elif not self.crossed_river:
                    reward += 5.0  # (+5.0)
            elif self.agent_pos[0] == self.river_row2:
                # Falling in the second river row
                if self.agent_pos[1] not in self.logs_row2:
                    reward = -150.0  # (-150 penalty)
                    terminated = True
                # Crossing the second river row
                elif not self.crossed_river:
                    reward += 10.0  # (+10.0)
                    self.crossed_river = True

        # Check goal
        if not terminated and self.agent_pos[0] == 0:
            reward = +150.0  # (+150 reward)
            terminated = True

        obs = self._get_obs()
        info = {}
        return obs, float(reward), bool(terminated), bool(truncated), info

    # Get the current observation
    def _get_obs(self):
        #Initialize grid
        grid = np.zeros((self.height, self.width), dtype=np.int8)

        # River and logs
        grid[self.river_row1, :] = 7
        grid[self.river_row2, :] = 7
        for c in self.logs_row1:
            grid[self.river_row1, c] = 8
        for c in self.logs_row2:
            grid[self.river_row2, c] = 8

        # Cars and trails
        for car in self.cars:
            for p in car["positions"]:
                # Prioritize car over trail for observation
                if grid[car["row"], p] != 1:
                    grid[car["row"], p] = 1
        for tr in self.trails:
            # Only show trail if not already a car or agent
            if grid[tr["row"], tr["col"]] not in [1, 2]:
                grid[tr["row"], tr["col"]] = 6

        # Goal and agent
        grid[0, :] = 3
        grid[self.agent_pos[0], self.agent_pos[1]] = 2

        # Local vision window
        v = self.vision
        half = v // 2
        ar, ac = self.agent_pos
        obs_window = np.zeros((v, v), dtype=np.int8)
        for i in range(v):
            for j in range(v):
                gr = ar + (i - half)
                gc = ac + (j - half)
                if 0 <= gr < self.height and 0 <= gc < self.width:
                    obs_window[i, j] = grid[gr, gc]

        # Extra observation: log positions and traffic lights
        obs_extra = np.zeros((4 + 1,), dtype=np.int8)
        obs_extra[0] = self.logs_row1[0] - ac
        obs_extra[1] = self.logs_row1[-1] - ac
        obs_extra[2] = self.logs_row2[0] - ac
        obs_extra[3] = self.logs_row2[-1] - ac
        obs_extra[0] = self.logs_row1[0] - ac
        obs_extra[1] = self.logs_row1[-1] - ac
        obs_extra[2] = self.logs_row2[0] - ac
        obs_extra[3] = self.logs_row2[-1] - ac

        # Find the first car lane traffic light (skip river rows)
        first_car_lane_light_idx = 0
        for r_idx in range(len(self.traffic_lights)): 
            r = r_idx + 1 
            if r not in [self.river_row1, self.river_row2]:
                first_car_lane_light_idx = r_idx
                break
        
        obs_extra[4] = self.traffic_lights[first_car_lane_light_idx]

        obs_flat = np.concatenate([obs_window.flatten(), obs_extra])
        return obs_flat


# Render the environment
    def render(self):

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.width * self.cell_size,
                                                   self.height * self.cell_size))
            pygame.display.set_caption("CrossTheRoad RL with Double River and Traffic Lights")
            self.clock = pygame.time.Clock()

            # Initialize car images (using placeholders if necessary)
            if not self.car_images:
                self.car_images = [
                    pygame.transform.scale(img.convert_alpha() if img.get_alpha() is not None else img.convert(),
                                           (self.cell_size, self.cell_size))
                    for img in self.raw_car_images
                ]
            # Initialize smoke images (using placeholders if necessary)
            if not self.smoke_images:
                self.smoke_images = [
                    pygame.transform.scale(img.convert_alpha() if img.get_alpha() is not None else img.convert(),
                                           (self.cell_size - 20, self.cell_size - 20))
                    for img in self.raw_smoke_images
                ]
            # Initialize wood images (using placeholders if necessary)
            if not self.wood_image and self.raw_wood_image:
                self.wood_image = pygame.transform.scale(
                    self.raw_wood_image.convert_alpha() if self.raw_wood_image.get_alpha() is not None else self.raw_wood_image.convert(),
                    (self.cell_size, self.cell_size))
            # Initialize water images (using placeholders if necessary)
            if not self.water_image and self.raw_water_image:
                self.water_image = pygame.transform.scale(
                    self.raw_water_image.convert_alpha() if self.raw_water_image.get_alpha() is not None else self.raw_water_image.convert(),
                    (self.width * self.cell_size, self.cell_size))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.window.fill((30, 40, 60))

        # Goal: checkered pattern
        small = 6
        for r in range(0, self.cell_size, small):
            for c in range(0, self.width * self.cell_size, small):
                color = (255, 255, 255) if (r // small + c // small) % 2 == 0 else (0, 0, 0)
                rect = pygame.Rect(c, r, small, small)
                pygame.draw.rect(self.window, color, rect)

        # Initialize lanes and river
        for r in range(1, self.height):
            if r in [self.river_row1, self.river_row2] and self.water_image:
                self.window.blit(self.water_image, (0, r * self.cell_size))
            else:
                lane_rect = pygame.Rect(0, r * self.cell_size, self.width * self.cell_size, self.cell_size)
                pygame.draw.rect(self.window, (90, 90, 110), lane_rect)

        # Initialize traffic lights and update their colors
        for r in range(1, self.height - 1):
            if r in [self.river_row1, self.river_row2]:
                continue
            color = (0, 255, 0) if self.traffic_lights[r - 1] == 1 else (255, 0, 0)
            pygame.draw.circle(self.window, color,
                               (self.width * self.cell_size - 10, r * self.cell_size + self.cell_size // 2), 8)

        #  Logs on the river
        if self.wood_image:
            for c in self.logs_row1:
                rect = self.wood_image.get_rect()
                rect.topleft = (c * self.cell_size, self.river_row1 * self.cell_size)
                self.window.blit(self.wood_image, rect)
            for c in self.logs_row2:
                rect = self.wood_image.get_rect()
                rect.topleft = (c * self.cell_size, self.river_row2 * self.cell_size)
                self.window.blit(self.wood_image, rect)

        #   Trails (smoke)
        for idx, tr in enumerate(self.trails):
            if not self.smoke_images: continue
            smoke_img = self.smoke_images[idx % len(self.smoke_images)]
            rect = smoke_img.get_rect()
            rect.topleft = (tr['col'] * self.cell_size + 10, tr['row'] * self.cell_size + 10)
            self.window.blit(smoke_img, rect)

        #   Cars
        for car in self.cars:
            if not self.car_images: continue
            img = self.car_images[car["img_index"]]
            if self.lanes_dir[car['row'] - 1] == 1:
                rotated = pygame.transform.rotate(img, -90)
            else:
                rotated = pygame.transform.rotate(img, 90)
            for p in car["positions"]:
                rect = rotated.get_rect()
                rect.topleft = (p * self.cell_size, car["row"] * self.cell_size)
                self.window.blit(rotated, rect)

        #   Agent
        a_rect = pygame.Rect(self.agent_pos[1] * self.cell_size + 8,
                             self.agent_pos[0] * self.cell_size + 8,
                             self.cell_size - 16, self.cell_size - 16)
        pygame.draw.ellipse(self.window, (80, 210, 120), a_rect)
        pygame.draw.circle(self.window, (255, 255, 255),
                           (a_rect.left + a_rect.width - 12, a_rect.top + 10), 4)

        pygame.display.flip()
        self.clock.tick(self.metadata.get("render_fps", 5))

    def close(self):
        if self.window:
            pygame.quit()
            self.window = None