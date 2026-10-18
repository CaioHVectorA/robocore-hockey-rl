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
  * Em curvas fechadas com alta aceleração centrípeta ($\omega > 6\text{ rad/s}$), o puck é ejetado lateralmente, exigindo que a rede neural aprenda dosagem fina de volante.
* **Cinemática Diferencial & Atuadores:**
  * Entre-eixos: $L_w = 14,5\text{ cm}$.
  * Inércia e atraso de resposta do motor DC + ponte H da placa Vespa modelados com $\tau_{\text{motor}} = 35\text{ ms}$.
  * Latência do protocolo de rádio Bluetooth BLE ($\sim 30\text{ ms}$) inserida no loop de simulação para evitar o choque de realidade no *Sim-to-Real*.

### 1.3. O Ringue Oficial com Cantos Arredondados
* Dimensões de mesa: **$180\text{ cm} \times 120\text{ cm}$**.
* **4 Cantos Arredondados com $R = 20\text{ cm}$:** Garantem que o puck deslize suavemente pelas bordas sem prender em quinas secas, permitindo jogadas de tabela circular.
* **Gols Embutidos:** Balizas de $35\text{ cm}$ de largura por $12\text{ cm}$ de profundidade nas extremidades.

---

## 2. Arquitetura da Codebase

```
robocore-hockey-rl/
├── README.md                   # Especificação física, matemática e guia Sim-to-Real
├── main.py                     # Launcher da interface gráfica interativa (Pygame)
├── train.py                    # Script de treinamento PPO e avaliação contra Bot
├── requirements.txt            # Dependências (numpy, pygame, gymnasium, torch, bleak)
├── core/
│   ├── constants.py            # Constantes 1:1 oficiais da RoboCore e do ringue
│   └── hockey_physics.py       # Motor de física com atrito de Coulomb e colisão com pá
├── bots/
│   └── hockey_bot.py           # Bot analítico especialista com interceptação preditiva
├── rl/
│   ├── env.py                  # Ambiente Gymnasium de 18 dimensões egocêntricas
│   └── ppo_trainer.py          # Treinador PPO contínuo Ator-Crítico
├── bridge/
│   └── vespa_ble_bridge.py     # Script Python pronto para conectar no HockeyBot real via BLE
└── ui/
    └── hockey_gui.py           # Interface visual rica com ringue, telemetria e joystick
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

## 4. Ponte Sim-to-Real: Como Controlar o Carrinho Físico via Bluetooth

O HockeyBot da RoboCore vem de fábrica com a placa **RoboCore Vespa (ESP32)** que já roda Bluetooth BLE e responde a comandos de velocidade das rodas.

O arquivo [`bridge/vespa_ble_bridge.py`](bridge/vespa_ble_bridge.py) já fornece a classe `VespaBluetoothBridge`:
1. Uma webcam posicionada sobre a mesa lê $(x, y, \theta)$ do robô e $(x, y)$ do puck a 60 FPS com OpenCV.
2. O vetor normalizado entra na rede neural `trainer.select_action(obs, deterministic=True)`.
3. A ação calculada $(u_L, u_R) \in [-1, 1]^2$ é enviada via BLE:
   ```python
   await bridge.send_wheel_speeds(u_left, u_right)
   ```
4. O robô real em pista executa exatamente as manobras e arremates aprendidos no simulador!
