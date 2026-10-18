"""
Constantes físicas oficiais do robô de hockey RoboCore (classe 1lb / HockeyBot / Falcon)
e especificações do ringue de hockey com cantos arredondados.
Todas as grandezas espaciais em centímetros (cm), massas em gramas (g) e tempo em segundos (s).
"""


class HockeyConstants:
    # --- DIMENSÕES DO RINGUE / ARENA ---
    ARENA_LENGTH: float = 180.0       # Comprimento total da arena (cm)
    ARENA_WIDTH: float = 120.0        # Largura total da arena (cm)
    HALF_LENGTH: float = 90.0         # X in [-90.0, +90.0] cm
    HALF_WIDTH: float = 60.0          # Y in [-60.0, +60.0] cm
    CORNER_RADIUS: float = 20.0       # Raio de curvatura dos 4 cantos arredondados (cm)

    # --- DIMENSÕES DOS GOLS ---
    GOAL_WIDTH: float = 35.0          # Largura da baliza (cm)
    HALF_GOAL_WIDTH: float = 17.5     # Y in [-17.5, +17.5] cm
    GOAL_DEPTH: float = 12.0          # Profundidade da rede (cm)
    GOAL_AREA_RADIUS: float = 25.0    # Semicírculo de proteção da pequena área

    # --- ESPECIFICAÇÕES DO CARRINHO FÍSICO ROBOCORE (HOCKEYBOT / CLASSE 1LB) ---
    ROBOT_WIDTH: float = 17.0         # Largura do chassi (17 cm conforme regulamento)
    ROBOT_LENGTH: float = 16.0        # Comprimento do chassi (16 cm conforme regulamento)
    HALF_ROBOT_W: float = 8.5         # Semi-largura
    HALF_ROBOT_L: float = 8.0         # Semi-comprimento
    ROBOT_HEIGHT: float = 5.0         # Altura do chassi
    ROBOT_MASS: float = 454.0         # Peso exato de 1 libra (454 gramas)
    WHEELBASE: float = 14.5           # Distância entre rodas de tração (cm)
    WHEEL_DIAMETER: float = 6.5       # Diâmetro das rodas de borracha
    MAX_WHEEL_SPEED: float = 85.0     # Velocidade linear máxima das rodas (~0.85 m/s)
    MOTOR_TAU: float = 0.035          # Inércia/atraso do motor DC + ponte H da Vespa (35 ms)

    # Pá frontal côncava (perfil de retenção do disco)
    SCOOP_DEPTH: float = 2.2          # Profundidade do rebaixo da pá frontal (cm)
    SCOOP_WIDTH: float = 11.0         # Largura útil da área da pá frontal (cm)

    # --- ESPECIFICAÇÕES DO PUCK (MINI PUCK DE HOCKEY OFICIAL) ---
    PUCK_DIAMETER: float = 5.0        # Diâmetro do disco oficial (5.0 cm)
    PUCK_RADIUS: float = 2.5          # Raio do disco (2.5 cm)
    PUCK_MASS: float = 38.0           # Massa aproximada do mini puck (38 g)
    PUCK_THICKNESS: float = 1.4       # Espessura (1.4 cm)

    # --- FÍSICA DE CONTATO E ATRITO SECO (COULOMB) ---
    GRAVITY: float = 981.0            # Aceleração gravitacional (cm/s^2)
    # Coeficiente de atrito cinético de deslizamento do disco no piso (MDF/vinil/acrílico)
    PUCK_SLIDING_FRICTION: float = 0.20
    PUCK_STOP_SPEED: float = 1.5      # Limite abaixo do qual o atrito estático trava o puck (cm/s)
    PUCK_WALL_RESTITUTION: float = 0.72 # Restituição elástica do puck nas paredes de acrílico/MDF
    ROBOT_PUCK_RESTITUTION: float = 0.65# Restituição no impacto com o chassi

    # Simulação a 60 FPS
    DT: float = 1.0 / 60.0
    MATCH_TIME_SECONDS: float = 180.0 # Partida oficial de 3 minutos
