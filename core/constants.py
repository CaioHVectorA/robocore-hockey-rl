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

    # --- REGRAS DE EQUIPE OFICIAIS (HOCKEY 3v3) ---
    ROBOTS_PER_TEAM: int = 3          # Partida oficial é sempre 3v3 (3 robôs por time, 6 na arena)
    ROLE_STRIKER: int = 0             # 0 = Atacante (busca o disco, conduz e arremata)
    ROLE_DEFENDER: int = 1            # 1 = Ala / Defensor (cobertura intermediária e apoio)
    ROLE_GOALIE: int = 2              # 2 = Goleiro (proteção da pequena área e rebatida)

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

    # --- PROPRIEDADES DE MASSA E INÉRCIA REAIS DERIVADAS DO CAD FUSION 360 ---
    # Limite oficial: 454.0 g (1.000 lb)
    ROBOT_MASS: float = 454.0

    # Posições de apoio no piso (cm em relação ao centro geométrico)
    AXLE_X_POS: float = -2.5           # Eixo das rodas tratoras
    SKID_X_POS: float = 6.8            # Patim frontal de apoio deslizante
    AXLE_SKID_DIST: float = 9.3        # Distância entre eixo e patim (6.8 - (-2.5))
    SKID_FRICTION_COEFF: float = 0.12  # Coeficiente de atrito do patim de Teflon/UHMW no piso
    COM_Z_HEIGHT: float = 2.38         # Altura do centro de gravidade em relação ao piso (cm)

    # Inércia rotacional I_zz calculada via Teorema dos Eixos Paralelos (Steiner) no CAD
    ROBOT_INERTIA: float = 13650.0      # Média geral (g*cm^2)
    ROBOT_INERTIA_STRIKER: float = 13787.0
    ROBOT_INERTIA_DEFENDER: float = 13483.0
    ROBOT_INERTIA_GOALIE: float = 13672.0

    # CoM longitudinal e fração de distribuição de carga estática nas rodas por papel
    COM_X_STRIKER: float = 0.024       # Avanço de massa para a pá frontal
    COM_X_DEFENDER: float = -0.237     # Centro neutro balanceado
    COM_X_GOALIE: float = -0.374       # Recuo de massa para as rodas de tração
    WHEEL_LOAD_STRIKER: float = 0.729  # 72.9% nas rodas, 27.1% no patim/pá
    WHEEL_LOAD_DEFENDER: float = 0.757 # 75.7% nas rodas, 24.3% no patim
    WHEEL_LOAD_GOALIE: float = 0.771   # 77.1% nas rodas, 22.9% no patim

    # --- ESPECIFICAÇÕES DO PUCK (MINI PUCK DE HOCKEY OFICIAL) ---
    PUCK_DIAMETER: float = 5.0        # Diâmetro do disco oficial (5.0 cm)
    PUCK_RADIUS: float = 2.5          # Raio do disco (2.5 cm)
    PUCK_MASS: float = 38.0           # Massa aproximada do mini puck (38 g)
    PUCK_THICKNESS: float = 1.4       # Espessura (1.4 cm)

    # --- FÍSICA DE CONTATO E ATRITO SECO (COULOMB) ---
    GRAVITY: float = 981.0            # Aceleração gravitacional (cm/s^2)
    # Coeficiente de atrito cinético de deslizamento do disco no piso (MDF/vinil/acrílico)
    PUCK_SLIDING_FRICTION: float = 0.20
    PUCK_STATIC_FRICTION: float = 0.28  # Atrito estático necessário para tirar o disco do repouso
    PUCK_STOP_SPEED: float = 1.5      # Limite abaixo do qual o atrito estático trava o puck (cm/s)
    PUCK_WALL_RESTITUTION: float = 0.72 # Restituição elástica do puck nas paredes de acrílico/MDF
    ROBOT_PUCK_RESTITUTION: float = 0.65# Restituição no impacto com o chassi

    # --- FIDELIDADE ELETROMECÂNICA DO CHASSI ROBOCORE (PLACA VESPA / ESP32) ---
    # Zona morta típica da ponte H e caixa de redução dos motores DC N20/25GA (~10-14% PWM mínimo)
    MOTOR_DEADBAND: float = 0.12
    # Coeficiente de atrito borracha-piso que limita a tração e gera derrapagem (wheel slip)
    TIRE_FRICTION_COEFF: float = 0.85
    TIRE_SLIP_FRICTION: float = 0.58  # Coeficiente cinético quando as rodas já estão derrapando
    # Aceleração angular crítica onde a força centrífuga ejeta o puck da pá frontal côncava
    SCOOP_EJECTION_OMEGA: float = 6.2  # rad/s (~355 graus/s)

    # --- CANAL DE COMUNICAÇÃO BLUETOOTH BLE (ESP32) & SIM-TO-REAL ---
    BLE_CONNECTION_INTERVAL: float = 0.030  # Intervalo médio de conexão BLE (~30 ms)
    BLE_JITTER_MAX: float = 0.012           # Jitter de transporte de rádio (+-12 ms)
    BLE_PACKET_LOSS_PROB: float = 0.025     # Taxa de perda/retransmissão de pacote em 2.4 GHz
    BLE_WATCHDOG_TIMEOUT: float = 0.200     # Timeout do watchdog no ESP32: corta motores se sem dados
    BLE_CONTROL_RATE_HZ: float = 30.0       # Taxa de envio de pacotes BLE ideal do ESP32 (30 Hz)

    # --- PIPELINE DE VISÃO ZENITAL PARCIAL & MARCAS NO CHASSI ---
    VISION_LATENCY_STEPS: int = 2           # Atraso de 2 quadros a 60 FPS (~33 ms de processamento)
    VISION_POS_NOISE_STD: float = 0.35      # Ruído de leitura de centróide da câmera (cm)
    VISION_ANGLE_NOISE_STD: float = 0.030   # Ruído de orientação angular (rad, ~1.7 graus)
    VISION_DROPOUT_PROB: float = 0.035      # Taxa de perda temporária de frame/tracking (3.5%)
    VISION_COLLISION_OCCLUSION_PROB: float = 0.15 # Chance de oclusão parcial em choques corpo a corpo
    VISION_PUCK_OCCLUSION_DIST: float = 11.0 # Distância na qual o disco entra sob a pá e pode sofrer oclusão parcial
    VISION_CONFIDENCE_DECAY: float = 0.88   # Decaimento de confiança em caso de perda de tracking

    # Especificações das marcas físicas fixadas no teto do chassi (17x16 cm)
    MARKER_FRONT_OFFSET: float = 4.5        # Ponto de proa/orientação frontal (cm)
    MARKER_REAR_OFFSET: float = -4.5        # Ponto de popa/papel tático (cm)
    MARKER_RADIUS: float = 1.8              # Raio do adesivo circular / tag visual no teto (cm)

    # Simulação a 60 FPS
    DT: float = 1.0 / 60.0
    MATCH_TIME_SECONDS: float = 180.0 # Partida oficial de 3 minutos
