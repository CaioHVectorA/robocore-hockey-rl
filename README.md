# RoboCore Hockey RL: Simulador Fiel & Aprendizado por Reforço (Classe 1lb)

> **Simulador Físico de Alta Fidelidade e Treinamento de RL (PPO)** modelando com rigor o **HockeyBot oficial da RoboCore (classe 1lb / 454 g, chassi 17x16 cm, placa Vespa ESP32 com Bluetooth BLE)**, a dinâmica real de **deslizamento do puck com atrito seco de Coulomb**, o ringue de hockey com **cantos arredondados de 20 cm** e a ponte direta para controle de robôs físicos via Python.

---

## 1. Por que este Simulador é Alinhado ao Carrinho Físico Real?

Diferente de simulações genéricas de futebol de botão (como HaxBall) ou robôs ideais:

### 1.1. Dinâmica do Disco (Puck) vs. Bola Esférica
* Uma bola de futebol rola acumulando momento angular tridimensional ($I\omega^2$).
* O **Mini Puck de Hockey** é um cilindro plano de borracha/plástico ($\varnothing 5,0\text{ cm}$, $38\text{ g}$) que **desliza no piso**:
  $$\vec{a}_{\text{puck}} = -\mu_k \, g \, \frac{\vec{v}}{\|\vec{v}\|}, \quad \text{com } \mu_k \approx 0.20$$
* **Travamento por Atrito Estático:** Quando a velocidade cai abaixo de $1,5\text{ cm/s}$, o disco **para completamente de uma vez**, impedindo micro-rolamentos infinitos irreais.

### 1.2. O Chassi Oficial da RoboCore (HockeyBot / Falcon)
* **Medidas Oficiais de Gabarito:** Largura máxima de **$17,0\text{ cm}$** $\times$ Comprimento de **$16,0\text{ cm}$** e peso oficial de **$454\text{ g}$** ($1\text{ lb}$).
* **Pá Frontal Côncava (*Scoop*):** A face frontal do chassi possui um rebaixo côncavo de $2,2\text{ cm}$ de profundidade e $11\text{ cm}$ de largura útil.
  * Em avanço retilíneo, o puck se acomoda no côncavo, permitindo que a IA **conduza o disco em alta velocidade**.
  * **Ejeção Centrífuga (Spin Shot):** Em giros bruscos ($\omega > 6,2\text{ rad/s}$), a aceleração centrífuga ejeta o puck da pá em disparo tangencial de alta velocidade, simulando o chute em rotação usado pelos pilotos reais.
* **Cinemática Diferencial & Atuadores Reais da Placa Vespa (ESP32):**
  * Entre-eixos: $L_w = 14,5\text{ cm}$.
  * **Zona Morta dos Motores (Deadband):** Motores DC N20/25GA com redução não se movem abaixo de ~12% de PWM devido ao atrito de partida. O simulador modela essa zona morta e a ponte BLE compensa o limiar físico.
  * **Limite de Tração & Derrapagem (Wheel Slip):** Acelerações ou frenagens que excedem a aderência borracha-piso ($\mu_{\text{pneu}} \approx 0,85$) geram derrapagem com redução de tração.
  * **Assimetria Estocástica de Motores:** Variação entre os motores esquerdo e direito sorteada por episódio para exigir controle em malha fechada realista.
  * **Canal Bluetooth BLE Real:** Latência média de $30\text{ ms}$, jitter de $\pm 12\text{ ms}$, taxa de perda de pacotes e watchdog de segurança do ESP32 ($200\text{ ms}$) que corta corrente se houver falha de sinal.
* **Percepção Sim-to-Real com Ruído de Câmera:**
  * O ambiente Gymnasium injeta ruído gaussiano nas posições e orientações para simular com perfeição uma câmera zenital (OpenCV / ArUco markers).

### 1.3. O Ringue Oficial com Cantos Arredondados
* Dimensões de mesa: **$180\text{ cm} \times 120\text{ cm}$**.
* **4 Cantos Arredondados com $R = 20\text{ cm}$:** Garantem que o puck deslize suavemente pelas bordas sem prender em quinas secas, permitindo jogadas de tabela circular.
* **Gols Embutidos:** Balizas de $35\text{ cm}$ de largura por $12\text{ cm}$ de profundidade nas extremidades.

### 1.4. Regra Oficial: Hockey de Robôs é Sempre 3v3 (6 Robôs na Arena)
Cada equipe oficial compete com **3 robôs HockeyBot** simultaneamente em campo:
* **Robô 0 (Atacante / Centro - Tag 'A'):** Ataca o puck, conduz no côncavo da pá frontal e finaliza a gol com disparos frontais ou *spin shots*.
* **Robô 1 (Ala / Defensor - Tag 'D'):** Contenção no meio-campo, fecha linhas de passe adversárias e dá suporte para rebotes.
* **Robô 2 (Goleiro - Tag 'G'):** Guarda a baliza, patrulha a pequena área e afasta o disco com a pá em investidas de perigo.

### 1.5. Percepção por Visão Computacional Parcial & Marcas no Chassi
Para viabilizar a transição **Sim-to-Real** sem a fragilidade de treinar RL ponta-a-ponta em pixels (que sofre com iluminação e alta latência >30 ms), o projeto adota a arquitetura desacoplada em 2 estágios (padrão oficial RoboCup SSL/VSS):

$$\text{Câmera Zenital 60 FPS} \xrightarrow{\text{Homografia + Marcas}} \text{Filtro de Kalman} \xrightarrow{\text{Vetor 27D}} \text{Política PPO } (<0.2\text{ ms}) \xrightarrow{\text{BLE 30 Hz}} \text{Vespa ESP32}$$

* **Marcas Físicas no Chassi (Goleiro, Ala e Atacante):**
  * Cada robô possui **duas marcas circulares ($\varnothing 3,6\text{ cm}$)** no teto ao longo do eixo longitudinal:
    1. **Marca de Proa (Frontal, $+4,5\text{ cm}$):** LED de alto brilho / ponto branco-amarelado que define sem ambiguidade o vetor diretor frontal $\vec{v}_{\theta} = (\cos\theta, \sin\theta)$.
    2. **Marca de Popa (Traseira, $-4,5\text{ cm}$):** Ponto colorido específico de cada papel tático:
       - **Atacante (A):** Ciano / Azul Claro (Time Azul) ou Laranja Intenso (Time Vermelho).
       - **Ala / Defensor (D):** Amarelo Dourado (Time Azul) ou Magenta (Time Vermelho).
       - **Goleiro (G):** Verde Lima (Time Azul) ou Roxo (Time Vermelho).
* **Visão Computacional Parcial Simulada (`SimulatedPartialVisionSensor`):**
  * Modela ruído gaussiano de centróide e calibração de homografia ($\sigma = 0,35\text{ cm}$).
  * **Oclusões Parciais em Choques Mecânicos:** Quando robôs se aglomeram a menos de $18\text{ cm}$, a probabilidade de perda de visão momentânea sobe para $15\%$.
  * **Oclusão do Puck no Scoop:** Quando o disco é retido na pá frontal côncava, a câmera zenital perde a borda do disco temporariamente.
  * **Filtro de Kalman Cinemático:** Suaviza o ruído das marcas e mantém a estimativa de trajetória e velocidade $(\hat{x}, \hat{y}, \hat{v}_x, \hat{v}_y)$ mesmo durante oclusões completas (com decaimento de confiança).
  * **Retículas Visuais no GUI:** Caixas verdes indicam rastreamento firme ($>85\%$), amarelas indicam predição pura por Kalman ($40-85\%$) e vermelhas indicam oclusão severa.

### 1.6. Modelagem CAD Paramétrica para Autodesk Fusion 360 & Dinâmica de Carga
* **Modelos CAD 3D Fidedignos:** Localizados em [`cad/models/`](cad/models/) nos formatos `.obj` e `.stl`, com script paramétrico [`cad/fusion_chassis_generator.py`](cad/fusion_chassis_generator.py) e script oficial da API do Fusion 360 [`cad/fusion360_build_chassis_script.py`](cad/fusion360_build_chassis_script.py).
* **Pesagem Rigorosa de 454.0 g (1 lb):** Lista de Materiais (BOM) completa de 13 itens totalizando exatamente 454,0 g (chassi base PETG/Nylon, placa Vespa ESP32, bateria LiPo, 2 motores DC micro metal, rodas de silicone com cubos de alumínio e latão, lâmina de desgaste em aço inox, parafusos Allen e patins de Teflon).
* **Tensor de Inércia & Centro de Massa (CoM):** Calculados componente a componente pelo Teorema dos Eixos Paralelos (Steiner):
  * **Atacante:** CoM avançado ($+0,024\text{ cm}$), $I_{zz} = 13.787\text{ g}\cdot\text{cm}^2$, $72,9\%$ de carga nas rodas e $27,1\%$ na pá (máxima retenção do puck).
  * **Ala / Defensor:** CoM neutro ($-0,237\text{ cm}$), $I_{zz} = 13.483\text{ g}\cdot\text{cm}^2$, $75,7\%$ nas rodas (equilíbrio e giros rápidos).
  * **Goleiro:** CoM recuado ($-0,374\text{ cm}$), $I_{zz} = 13.672\text{ g}\cdot\text{cm}^2$, $77,1\%$ nas rodas (máxima tração lateral no gol).
* **Transferência Dinâmica de Carga:** O motor físico modela a transferência longitudinal $\Delta N_x = m \cdot a_x \cdot \frac{h_{CoM}}{L}$, limite de tração instantâneo e arrasto de atrito do patim dianteiro de Teflon ($\mu_{\text{skid}} = 0,12$).
* **Documentação Técnica Completa:** Veja [`docs/CHASSIS_CAD_PHYSICS_SPEC.md`](docs/CHASSIS_CAD_PHYSICS_SPEC.md).

---

## 2. Arquitetura da Codebase

```
robocore-hockey-rl/
├── README.md                   # Visão geral, especificações físicas e guias de execução
├── main.py                     # Launcher da central gráfica interativa (Pygame)
├── train.py                    # Script de treinamento PPO Multi-Agente 3v3
├── requirements.txt            # Dependências (numpy, pygame, gymnasium, torch, bleak, opencv-python)
├── cad/
│   ├── fusion_chassis_generator.py   # Gerador paramétrico de malhas 3D (OBJ/STL) e cálculo analítico de CoM e inércia
│   ├── fusion360_build_chassis_script.py # Script oficial para execução na API do Autodesk Fusion 360
│   └── models/
│       ├── chassis_atacante.obj / .stl # Modelo 3D oficial do Atacante (Falcon Striker)
│       ├── chassis_ala.obj / .stl      # Modelo 3D oficial do Ala (Falcon Defender)
│       ├── chassis_goleiro.obj / .stl  # Modelo 3D oficial do Goleiro (Falcon Goalie)
│       └── chassis_specs.json          # Metadados físicos consolidados (BOM, CoM, Izz e cargas)
├── docs/
│   └── CHASSIS_CAD_PHYSICS_SPEC.md     # Manual completo de engenharia mecânica, BOM, Steiner e Fusion 360
├── core/
│   ├── constants.py            # Constantes 1:1 oficiais (arena, puck, 3v3, marcas de chassi e inércia CAD)
│   └── hockey_physics.py       # Motor físico 2D (Coulomb, deadband, slip, scoop 3v3, canal BLE e transferência de carga)
├── bots/
│   └── hockey_bot.py           # Bot analítico especialista com papéis táticos (Atacante, Ala, Goleiro)
├── vision/
│   └── camera_tracker.py       # Homografia, detecção de marcas, Kalman e SimulatedPartialVisionSensor
├── rl/
│   ├── env.py                  # Ambiente Gymnasium Multi-Agente 3v3 integrado à visão parcial
│   └── ppo_trainer.py          # Treinador PPO Multi-Agente com inferência em lote [3, 2]
├── bridge/
│   └── vespa_ble_bridge.py     # VespaBluetoothBridge e VespaTeamBridge (controle 3v3 BLE a 30 Hz)
└── ui/
    └── hockey_gui.py           # Central visual com 6 robôs, marcas no chassi, retículas de visão e telemetria
```

---

## 3. Como Executar

### 3.1. Central Gráfica Interativa
```bash
python main.py
```

**Recursos Interativos da Interface:**
* **Velocidade Acelerada ($1\times$ a $100\times$):** Acompanhe o aprendizado visual em segundos.
* **Botão "Ficar Burro (Reset IA)":** Reinicia os pesos neurais para rever a evolução do zero.
* **Telemetria de Hardware:** Exibe em tempo real latência do BLE, estado do Scoop, aviso de derrapagem (*Wheel Slip*) e zona morta.
* **Interação com Mouse na Bancada:**
  * **Clique Esquerdo + Arrastar:** Mova o puck ou os robôs livremente pela arena em tempo real.
  * **Clique Direito + Puxar ("Peteleco"):** Aplica um disparo de velocidade no puck com vetor ajustável.
* **Modo "App Joystick" (WASD / Setas):**
  * Pilote o robô no teclado com a mesma dinâmica do app *RoboCore Joystick* (W/S avança e dá ré, A/D esterça, Espaço dá giro rápido no eixo).
  * Jogue você mesmo no teclado contra o seu agente de RL ou contra o Bot heurístico!

### 3.2. Treinamento via Linha de Comando (Headless / Benchmark)
```bash
python train.py --steps 30000 --save hockey_ppo.pt
```

---

## 4. Ponte Sim-to-Real: Comunicação Bluetooth com o Carrinho Físico

O HockeyBot da RoboCore utiliza a placa **RoboCore Vespa (ESP32)** que opera via Bluetooth Low Energy (BLE).

O arquivo [`bridge/vespa_ble_bridge.py`](bridge/vespa_ble_bridge.py) fornece ferramentas completas de comunicação:

### 4.1. Escanear Robôs no Alcance
```bash
python -m bridge.vespa_ble_bridge --scan
```

### 4.2. Autoteste de Bancada do Hardware Físico
Testa as rodas e o giro rápido (spin shot) em bancada com parada segura:
```bash
python -m bridge.vespa_ble_bridge --test
```

### 4.3. Calibração de Motores (Trim)
Se o robô físico tender ligeiramente para a esquerda ou direita devido a tolerâncias mecânicas:
```bash
python -m bridge.vespa_ble_bridge --test --trim-left 1.0 --trim-right 0.95
```

### 4.4. Controle em Pista com Visão Computacional (Equipe 3v3)
1. Uma webcam posicionada sobre a mesa lê $(x, y, \theta)$ de cada robô e $(x, y)$ do puck a 60 FPS com OpenCV.
2. Cada robô gera seu vetor egocêntrico de 27 dimensões `obs[i]`.
3. A rede neural calcula as ações em lote `acts = trainer.select_action(team_obs, deterministic=True)`.
4. A classe `VespaTeamBridge` despacha as ações em paralelo via BLE a 30 Hz com compensação de deadband e trim:
   ```python
   await team_bridge.send_team_actions(acts, apply_compensation=True)
   ```
5. Os 3 robôs reais em pista executam coordenadamente as manobras, coberturas defensivas e arremates treinados no simulador!
