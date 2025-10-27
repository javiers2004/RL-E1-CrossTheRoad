import pickle
import numpy as np
from env import CrossTheRoadVisionEnv
import cv2  # Para escribir el vídeo
import pygame # <--- Importamos pygame aquí
import pygame.surfarray # <--- Importamos surfarray aquí

# QFILE = "q_table.pkl"
# QFILE = "sarsa_table.pkl"
QFILE = "expected_sarsa_table.pkl"

# --- Configuración del vídeo ---
VIDEO_FILENAME = "partida_agente.mp4"
# TU env.py tiene un límite de 5 FPS (self.clock.tick(5))
# Por lo tanto, grabamos a 5 FPS para que la velocidad sea normal.
FPS = 5
# -----------------------------


class PolicyPlayer:
    def __init__(self, qpath=QFILE):
        self.q = {}
        if qpath:
            try:
                with open(qpath, "rb") as f:
                    self.q = pickle.load(f)
                print("Tabla Q cargada.")
            except Exception as e:
                print("No se pudo cargar la tabla Q:", e)
                self.q = {}

    def act(self, obs):
        key = tuple(obs.flatten().tolist())
        if key in self.q:
            return int(np.argmax(self.q[key]))
        return np.random.randint(0, 4)

# Ejemplo de grabación
if __name__ == "__main__":
    
    # 1. Inicializamos el entorno (sin modificarlo)
    env = CrossTheRoadVisionEnv(
        height=14, 
        width=12, 
        vision=3, 
        car_spawn_prob=0.2, 
        max_cars_per_lane=1, 
        trail_prob=0.2
    )
    
    player = PolicyPlayer()

    # 2. Hacemos un 'reset' y un 'render' inicial
    # Esto es OBLIGATORIO para que el env cree la ventana (self.window)
    print("Inicializando entorno y ventana...")
    obs, info = env.reset()
    env.render() # <-- Esta llamada crea 'env.window'
    
    # 3. Obtenemos el tamaño directamente de la ventana de Pygame
    try:
        width, height = env.window.get_size()
    except AttributeError:
        print("Error: env.window no se inicializó.")
        print("Asegúrate de que env.render() funciona correctamente por sí solo.")
        env.close()
        exit()
    
    # 4. Inicializamos el VideoWriter de OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(VIDEO_FILENAME, fourcc, FPS, (width, height))

    print(f"Grabando 10 episodios en {VIDEO_FILENAME} a {FPS} FPS...")
    print("Una ventana de Pygame aparecerá. Es necesario para la captura.")

    # 5. Bucle de los 10 episodios (grabando en un solo vídeo)
    for ep in range(10):
        obs, info = env.reset()
        done = False
        while not done:
            # Obtenemos la acción del agente
            action = player.act(obs)
            
            # Damos el paso en el entorno
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Renderizamos el fotograma actual EN LA VENTANA
            env.render() # <-- Esto dibuja en la ventana oculta/visible
            
            # --- Magia: Capturamos la ventana ---
            
            # 1. Obtenemos la superficie (la ventana) de Pygame desde el 'env'
            frame_pygame_surface = env.window 
            
            # 2. Convertimos esa superficie en un array de numpy
            frame_pygame = pygame.surfarray.array3d(frame_pygame_surface)
            
            # 3. Corregimos la orientación (Pygame es W,H,C -> NumPy es H,W,C)
            frame_rgb = np.transpose(frame_pygame, (1, 0, 2))
            
            # 4. Convertimos de RGB a BGR (formato de OpenCV)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            # 5. Escribimos el fotograma en el archivo de vídeo
            video_writer.write(frame_bgr)
            
            # Ya no necesitamos time.sleep, porque env.render() ya
            # incluye un clock.tick(5) que limita los FPS.
            
    # 6. Cerramos todo
    video_writer.release()  # ¡Muy importante! Guarda y cierra el archivo de vídeo
    env.close()
    
    print(f"¡Grabación completa! Vídeo guardado en: {VIDEO_FILENAME}")