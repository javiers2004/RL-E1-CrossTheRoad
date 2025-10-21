import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pygame


class CrossTheRoadVisionEnv(gym.Env):
    """
    CrossTheRoad-like environment with local vision window, hazards, a double-wide river with moving logs,
    and traffic lights controlling car movement.
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

    def __init__(self, height=14, width=12, vision=3,
                 car_spawn_prob=0.2, max_cars_per_lane=2,
                 trail_prob=0.2, seed=None):
        super().__init__()
        assert vision % 2 == 1 and vision >= 3
        self.height = height
        self.width = width
        self.vision = vision
        self.car_spawn_prob = car_spawn_prob
        self.max_cars_per_lane = max_cars_per_lane
        self.trail_prob = trail_prob
        self.cell_size = 48

        self.action_space = spaces.Discrete(4)
        # Observation space: vision window + log positions + traffic light states
        self.num_car_lanes = self.height - 3  # Exclude goal, river_row1, river_row2
        self.observation_space = spaces.Box(low=0, high=8,
                                            shape=(vision * vision + 4 + self.num_car_lanes,),
                                            dtype=np.int8)

        self.window = None
        self.clock = None
        self.random = random.Random(seed)
        self.np_random = np.random.RandomState(seed)

        self.lanes_dir = [self.random.choice([-1, 1]) for _ in range(1, self.height - 1)]

        # Load car images (Assuming images/ directory structure)
        import os
        base_path = os.path.dirname(os.path.abspath(__file__))

        # --- PATH CORRECTION: Use placeholders if images aren't available for execution ---
        # NOTE: If you run this code, ensure 'images' directory exists with car/smoke/log/water images.
        def load_img(path):
            try:
                return pygame.image.load(path)
            except pygame.error:
                # Placeholder image if file not found
                print(f"Warning: Image not found at {path}. Using placeholder.")
                return pygame.Surface((48, 48), pygame.SRCALPHA)

        self.raw_car_images = []
        for i in range(1, 6):
            img_path = os.path.join(base_path, "images", f"car{i}.png")
            self.raw_car_images.append(load_img(img_path))

        self.raw_smoke_images = []
        for i in range(1, 5):
            img_path = os.path.join(base_path, "images", f"smoke{i}.png")
            self.raw_smoke_images.append(load_img(img_path))

        self.raw_wood_image = load_img(os.path.join(base_path, "images", "log.png"))
        self.raw_water_image = load_img(os.path.join(base_path, "images", "water.png"))
        # --------------------------------------------------------------------------------

        self.car_images = None
        self.smoke_images = None
        self.wood_image = None
        self.water_image = None

        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.random.seed(seed)
            self.np_random.seed(seed)

        self.agent_pos = [self.height - 1, self.width // 2]
        self.cars = []
        self.trails = []
        self.t = 0
        self.crossed_river = False

        # Double-wide river: two consecutive rows
        self.river_row1 = self.random.randint(1, self.height - 3)
        self.river_row2 = self.river_row1 + 1
        log_start1 = self.random.randint(0, self.width - 3)
        log_start2 = self.random.randint(0, self.width - 3)
        self.logs_row1 = list(range(log_start1, log_start1 + 3))
        self.logs_row2 = list(range(log_start2, log_start2 + 3))

        # Traffic lights for car lanes (0=red, 1=green)
        self.traffic_lights = [1 for _ in range(1, self.height - 1)]
        self.traffic_light_timer = 0
        self.traffic_light_interval = 5

        return self._get_obs(), {}

    def step(self, action):
        self.t += 1
        reward = 0.0
        terminated = False
        truncated = False

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
        # old_col = self.agent_pos[1] # Not used for car collision pre-check in the original code

        # Agent movement
        if action == 0 and self.agent_pos[0] > 0:  # up
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.height - 1:  # down
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:  # left
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.width - 1:  # right
            self.agent_pos[1] += 1

        # --- CORRECCIÓN DE RECOMPENSAS ---
        if self.agent_pos[0] < old_row:
            reward += 1.0  # Avanzar
        elif self.agent_pos[0] > old_row:
            reward -= 0.5  # Retroceder (menos penalización)
        else:
            reward -= 0.1  # Espera/Lateral
        # ---------------------------------

        # Move logs
        self.logs_row1 = [(c + 1) % self.width for c in self.logs_row1]
        self.logs_row2 = [(c - 1) % self.width for c in self.logs_row2]

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

                    # --- CHECK for collision with moving car ---
                    if car['row'] == self.agent_pos[0]:
                        # Colisión: Agente está en la nueva posición del coche O agente y coche se cruzaron (no es perfecto, pero sirve)
                        if next_p == self.agent_pos[1] or (p == self.agent_pos[1] and next_p == self.agent_pos[1]):
                            reward = -150.0  # Gran penalización
                            terminated = True

                    if car.get('trail', False):
                        new_trails.append({'row': car['row'], 'col': p, 'ttl': 5})
                    if 0 <= next_p < self.width:
                        new_positions.append(next_p)
                car['positions'] = new_positions
            else:
                # --- CHECK for collision with stationary car (Red light) ---
                for p in car['positions']:
                    if car['row'] == self.agent_pos[0] and p == self.agent_pos[1]:
                        reward = -150.0  # Gran penalización
                        terminated = True
                        break

        self.trails.extend(new_trails)
        self.cars = [car for car in self.cars if car['positions']]

        # Spawn new cars (same logic as before)
        for r in range(1, self.height - 1):
            if r in [self.river_row1, self.river_row2]:
                continue
            dir = self.lanes_dir[r - 1]
            cars_in_row = [car for car in self.cars if car['row'] == r]
            n_existing = len(cars_in_row)
            if n_existing < self.max_cars_per_lane and self.random.random() < self.car_spawn_prob:
                spawn_col = 0 if dir == 1 else self.width - 1
                occupied = {p for car in cars_in_row for p in car['positions']}
                if spawn_col not in occupied:
                    self.cars.append({
                        'row': r,
                        'positions': [spawn_col],
                        'trail': self.random.random() < self.trail_prob,
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
                    reward = -150.0  # Gran penalización
                    terminated = True
                    break

        # Check river and logs
        if not terminated:
            if self.agent_pos[0] == self.river_row1:
                if self.agent_pos[1] not in self.logs_row1:
                    reward = -150.0  # Gran penalización
                    terminated = True
                elif not self.crossed_river:
                    reward += 5.0
            elif self.agent_pos[0] == self.river_row2:
                if self.agent_pos[1] not in self.logs_row2:
                    reward = -150.0  # Gran penalización
                    terminated = True
                elif not self.crossed_river:
                    reward += 10.0
                    self.crossed_river = True

        # Check goal
        if not terminated and self.agent_pos[0] == 0:
            reward = +150.0  # Gran recompensa
            terminated = True

        obs = self._get_obs()
        info = {}
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _get_obs(self):
        # ... (Same logic as before) ...
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
        obs_extra = np.zeros((4 + self.num_car_lanes,), dtype=np.int8)
        obs_extra[0] = self.logs_row1[0] - ac
        obs_extra[1] = self.logs_row1[-1] - ac
        obs_extra[2] = self.logs_row2[0] - ac
        obs_extra[3] = self.logs_row2[-1] - ac
        car_lane_idx = 0
        for r in range(1, self.height - 1):
            if r not in [self.river_row1, self.river_row2]:
                obs_extra[4 + car_lane_idx] = self.traffic_lights[r - 1]
                car_lane_idx += 1

        obs_flat = np.concatenate([obs_window.flatten(), obs_extra])
        return obs_flat

    # ... (render and close methods omitted for brevity, they remain unchanged) ...

    def render(self):
        # ... (render logic remains the same) ...
        # (Included for completeness, but assuming it's in the original env.py)
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.width * self.cell_size,
                                                   self.height * self.cell_size))
            pygame.display.set_caption("CrossTheRoad RL with Double River and Traffic Lights")
            self.clock = pygame.time.Clock()

            # Initialize images (using placeholders if necessary)
            if not self.car_images:
                self.car_images = [
                    pygame.transform.scale(img.convert_alpha() if img.get_alpha() is not None else img.convert(),
                                           (self.cell_size, self.cell_size))
                    for img in self.raw_car_images
                ]
            if not self.smoke_images:
                self.smoke_images = [
                    pygame.transform.scale(img.convert_alpha() if img.get_alpha() is not None else img.convert(),
                                           (self.cell_size - 20, self.cell_size - 20))
                    for img in self.raw_smoke_images
                ]
            if not self.wood_image and self.raw_wood_image:
                self.wood_image = pygame.transform.scale(
                    self.raw_wood_image.convert_alpha() if self.raw_wood_image.get_alpha() is not None else self.raw_wood_image.convert(),
                    (self.cell_size, self.cell_size))
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

        # Lanes and river
        for r in range(1, self.height):
            # lane_rect = pygame.Rect(0, r * self.cell_size, self.width * self.cell_size, self.cell_size)
            if r in [self.river_row1, self.river_row2] and self.water_image:
                self.window.blit(self.water_image, (0, r * self.cell_size))
            else:
                lane_rect = pygame.Rect(0, r * self.cell_size, self.width * self.cell_size, self.cell_size)
                pygame.draw.rect(self.window, (90, 90, 110), lane_rect)

        # Traffic lights
        for r in range(1, self.height - 1):
            if r in [self.river_row1, self.river_row2]:
                continue
            color = (0, 255, 0) if self.traffic_lights[r - 1] == 1 else (255, 0, 0)
            pygame.draw.circle(self.window, color,
                               (self.width * self.cell_size - 10, r * self.cell_size + self.cell_size // 2), 8)

        # Logs
        if self.wood_image:
            for c in self.logs_row1:
                rect = self.wood_image.get_rect()
                rect.topleft = (c * self.cell_size, self.river_row1 * self.cell_size)
                self.window.blit(self.wood_image, rect)
            for c in self.logs_row2:
                rect = self.wood_image.get_rect()
                rect.topleft = (c * self.cell_size, self.river_row2 * self.cell_size)
                self.window.blit(self.wood_image, rect)

        # Car trails
        for idx, tr in enumerate(self.trails):
            if not self.smoke_images: continue
            smoke_img = self.smoke_images[idx % len(self.smoke_images)]
            rect = smoke_img.get_rect()
            rect.topleft = (tr['col'] * self.cell_size + 10, tr['row'] * self.cell_size + 10)
            self.window.blit(smoke_img, rect)

        # Cars
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

        # Agent
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