import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pygame

class FroggerVisionEnv(gym.Env):
    """
    Frogger-like environment with local vision window, hazards, and a river.
    Grid values:
      0 = empty
      1 = car
      2 = agent
      3 = goal row
      4 = meteor warning
      5 = meteor impact (danger)
      6 = car trail (danger)
      7 = river (lethal unless on bridge)
      8 = bridge (safe)
    """
    metadata = {"render_modes": ["human"], "render_fps": 3}

    def __init__(self, height=14, width=12, vision=3,
                car_spawn_prob=0.2, meteor_prob=0.2,
                trail_prob=0.2, seed=None):
        super().__init__()
        assert vision % 2 == 1 and vision >= 3
        self.height = height
        self.width = width
        self.vision = vision
        self.car_spawn_prob = car_spawn_prob
        self.meteor_prob = meteor_prob
        self.trail_prob = trail_prob
        self.cell_size = 48

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=8,
                                            shape=(vision * vision + 2,),
                                            dtype=np.int8)

        self.window = None
        self.clock = None
        self.random = random.Random(seed)
        self.np_random = np.random.RandomState(seed)

        self.lanes_dir = [self.random.choice([-1, 1]) for _ in range(1, self.height - 1)]

        # Cargar imágenes de coches sin convertir (convert_alpha en render)
        import os
        base_path = os.path.dirname(__file__)
        self.raw_car_images = []
        for i in range(1, 6):
            img_path = os.path.join(base_path, "images", f"car{i}.png")
            self.raw_car_images.append(pygame.image.load(img_path))
        self.car_images = None  # se inicializan en render()

        # Humo para los rastros
        self.raw_smoke_images = []
        for i in range(1, 5):
            img_path = os.path.join(base_path, "images", f"smoke{i}.png")
            self.raw_smoke_images.append(pygame.image.load(img_path))
        self.smoke_images = None  # se inicializan en render()

        # Puente (madera)
        self.raw_wood_image = pygame.image.load(os.path.join(base_path, "images", "wood.png"))
        self.wood_image = None  # se inicializa en render()

        # Río (agua)
        self.raw_water_image = pygame.image.load(os.path.join(base_path, "images", "water.png"))
        self.water_image = None  # se inicializa en render()

        # Meteoros / explosiones
        self.raw_explosion_images = []
        for i in range(1, 3):  # explosion1.png y explosion2.png
            self.raw_explosion_images.append(
                pygame.image.load(os.path.join(base_path, "images", f"explosion{i}.png"))
            )
        self.explosion_images = None  # se inicializa en render()

        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.random.seed(seed)
            self.np_random.seed(seed)

        self.agent_pos = [self.height - 1, self.width // 2]
        self.cars = []
        self.trails = []
        self.meteors = []
        self.t = 0
        self.crossed_bridge = False 

        # --- NUEVO: río y puente ---
        self.river_row = self.random.randint(1, self.height - 2)
        bridge_start = self.random.randint(0, self.width - 3)
        self.bridge_cols = list(range(bridge_start, bridge_start + 3))

        return self._get_obs(), {}


    def step(self, action):
        self.t += 1
        reward = 0.0
        terminated = False
        truncated = False
        old_row = self.agent_pos[0]

        # --- Movimiento del agente ---
        if action == 0 and self.agent_pos[0] > 0:  # up
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.height - 1:  # down
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:  # left
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.width - 1:  # right
            self.agent_pos[1] += 1

        # --- Recompensas de movimiento ---
        if self.agent_pos[0] < old_row:
            reward += 1.0   # avanzar hacia la meta
        elif self.agent_pos[0] > old_row:
            reward -= 1.1   # retroceder
        else:
            reward -= 0.1  # movimiento lateral o quedarse quieto (ligero coste)

        # --- Mover coches y generar rastros ---
        new_trails = []
        for car in self.cars:
            row_idx = car['row'] - 1
            dir = self.lanes_dir[row_idx]
            new_positions = []
            for p in car['positions']:
                next_p = p + dir

                # Colisión con el agente
                if car['row'] == self.agent_pos[0]:
                    if min(p, next_p) <= self.agent_pos[1] <= max(p, next_p):
                        reward = -20.0
                        terminated = True

                # Generar rastro si corresponde
                if car.get('trail', False):
                    new_trails.append({'row': car['row'], 'col': p, 'ttl': 5})

                if 0 <= next_p < self.width:
                    new_positions.append(next_p)

            car['positions'] = new_positions

        self.trails.extend(new_trails)
        self.cars = [car for car in self.cars if car['positions']]

        # --- Generar nuevos coches probabilísticamente ---
        for r in range(1, self.height - 1):
            # no generar coches en el río
            if r == self.river_row:
                continue

            # verificar si hay coches actualmente en esta fila
            cars_in_row = [car for car in self.cars if car['row'] == r]

            # si no hay coches en la fila, aplicar probabilidad de spawn
            if not cars_in_row:
                if self.random.random() < self.car_spawn_prob:  # ejemplo: 0.1 = 10%
                    pos = 0 if self.lanes_dir[r - 1] == 1 else self.width - 1
                    self.cars.append({
                        'row': r,
                        'positions': [pos],
                        'trail': self.random.random() < self.trail_prob,
                        'img_index': self.random.randint(0, len(self.raw_car_images) - 1)
                    })


        # --- Meteoritos ---
        if self.random.random() < self.meteor_prob:
            r = self.random.randint(1, self.height - 2)
            c = self.random.randint(0, self.width - 1)
            self.meteors.append({'row': r, 'col': c, 'ttl': 3, 'stage': 'warning'})

        # Actualizar meteoritos
        new_meteors = []
        for m in self.meteors:
            m['ttl'] -= 1
            if m['ttl'] <= 0:
                if m['stage'] == 'warning':
                    m['stage'] = 'impact'
                    m['ttl'] = 2
                    new_meteors.append(m)
                else:
                    continue
            else:
                new_meteors.append(m)
        self.meteors = new_meteors

        # Actualizar rastros
        for tr in self.trails:
            tr['ttl'] -= 1
        self.trails = [t for t in self.trails if t['ttl'] > 0]

        # --- Colisiones con peligros ---
        if not terminated:
            for tr in self.trails:
                if tr["row"] == self.agent_pos[0] and tr["col"] == self.agent_pos[1]:
                    reward = -20.0
                    terminated = True
                    break

        if not terminated:
            for m in self.meteors:
                if m['stage'] == 'impact':
                    if m['row'] == self.agent_pos[0] and m['col'] == self.agent_pos[1]:
                        reward = -20.0
                        terminated = True
                        break
        
        # --- Revisión de río ---
        if self.agent_pos[0] == self.river_row:
            if self.agent_pos[1] not in self.bridge_cols:
                reward = -20.0
                terminated = True
            elif not self.crossed_bridge:
                reward += 10.0
                self.crossed_bridge = True
            

        # --- Llegar a la meta ---
        if not terminated and self.agent_pos[0] == 0:
            reward = +30.0
            terminated = True

        # --- Resultado final ---
        obs = self._get_obs()
        info = {}
        return obs, float(reward), bool(terminated), bool(truncated), info


    def _get_obs(self):
        grid = np.zeros((self.height, self.width), dtype=np.int8)

        # río y puente
        grid[self.river_row, :] = 7
        for c in self.bridge_cols:
            grid[self.river_row, c] = 8

        # coches y rastros
        for car in self.cars:
            for p in car["positions"]:
                grid[car["row"], p] = 1
        for tr in self.trails:
            grid[tr["row"], tr["col"]] = 6
        for m in self.meteors:
            if m['stage'] == 'warning':
                grid[m['row'], m['col']] = 4
            elif m['stage'] == 'impact':
                grid[m['row'], m['col']] = 5

        # meta y agente
        grid[0, :] = 3
        grid[self.agent_pos[0], self.agent_pos[1]] = 2

        # --- Ventana local centrada en el agente ---
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

        # --- Vector extra: posición relativa del puente ---
        obs_extra = np.zeros((2,), dtype=np.int8)
        obs_extra[0] = self.bridge_cols[0] - ac  # inicio del puente relativo al agente
        obs_extra[1] = self.bridge_cols[-1] - ac  # fin del puente relativo al agente

        # --- Concatenar todo en un array 1D ---
        obs_flat = np.concatenate([obs_window.flatten(), obs_extra])
        return obs_flat

    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.width * self.cell_size,
                                                self.height * self.cell_size))
            pygame.display.set_caption("Frogger RL con Río y Puente")
            self.clock = pygame.time.Clock()

            # Convertir y escalar imágenes ahora que display está inicializado
            self.car_images = [
                pygame.transform.scale(img.convert_alpha(), (self.cell_size, self.cell_size))
                for img in self.raw_car_images
            ]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.window.fill((30, 40, 60))

        # --- Meta: mosaico de cuadritos negros y blancos ---
        small = 6  # tamaño de cuadrito
        for r in range(0, self.cell_size, small):
            for c in range(0, self.width * self.cell_size, small):
                color = (255, 255, 255) if (r // small + c // small) % 2 == 0 else (0, 0, 0)
                rect = pygame.Rect(c, r, small, small)
                pygame.draw.rect(self.window, color, rect)

        # --- Carriles ---
        for r in range(1, self.height):
            # Color de fondo de carril o río
            lane_rect = pygame.Rect(0, r * self.cell_size, self.width * self.cell_size, self.cell_size)
            
            if r == self.river_row:
                # Dibujar agua con sprite
                if self.water_image is None:
                    self.water_image = pygame.transform.scale(self.raw_water_image.convert_alpha(),
                                                            (self.width * self.cell_size, self.cell_size))
                self.window.blit(self.water_image, (0, r * self.cell_size))
            else:
                # Color normal de carril
                color = (90, 90, 110)
                pygame.draw.rect(self.window, color, lane_rect)


        # --- Puente ---
        if self.wood_image is None:
            self.wood_image = pygame.transform.scale(self.raw_wood_image.convert_alpha(),
                                                    (self.cell_size, self.cell_size))


        for c in self.bridge_cols:
            rect = self.wood_image.get_rect()
            rect.topleft = (c * self.cell_size, self.river_row * self.cell_size)
            self.window.blit(self.wood_image, rect)

        # --- Rastro de coches con sprites de humo ---
        if self.smoke_images is None:
            # escalar imágenes de humo
            self.smoke_images = [
                pygame.transform.scale(img.convert_alpha(), (self.cell_size - 20, self.cell_size - 20))
                for img in self.raw_smoke_images
            ]

        for idx, tr in enumerate(self.trails):
            # Asignar sprite cíclico según índice del rastro
            smoke_img = self.smoke_images[idx % len(self.smoke_images)]
            rect = smoke_img.get_rect()
            rect.topleft = (tr['col'] * self.cell_size + 10, tr['row'] * self.cell_size + 10)
            self.window.blit(smoke_img, rect)


        # --- Coches con sprite fijo y rotado según dirección ---
        for car in self.cars:
            img = self.car_images[car["img_index"]]
            if self.lanes_dir[car['row'] - 1] == 1:
                rotated = pygame.transform.rotate(img, -90)  # derecha
            else:
                rotated = pygame.transform.rotate(img, 90)   # izquierda
            for p in car["positions"]:
                rect = rotated.get_rect()
                rect.topleft = (p * self.cell_size, car["row"] * self.cell_size)
                self.window.blit(rotated, rect)

        # --- Meteoritos ---
        for m in self.meteors:
            cx, cy = m['col'] * self.cell_size + self.cell_size // 2, m['row'] * self.cell_size + self.cell_size // 2
            if m['stage'] == 'warning':
                pygame.draw.circle(self.window, (255, 255, 100), (cx, cy), 10)
            elif m['stage'] == 'impact':
                pygame.draw.circle(self.window, (255, 80, 0), (cx, cy), 25)

        # --- Agente ---
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
