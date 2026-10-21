# Hockey Lab — simulação, PPO e visão zenital

Simulador 2D de hockey 3v3 com contatos de corpos rígidos, controle diferencial e uma arquitetura compartilhada entre percepção simulada e câmera real. O projeto é uma plataforma de desenvolvimento: parâmetros de hardware ainda precisam ser medidos e uma política precisa ser treinada e avaliada antes de uso nos robôs.

## Instalar e jogar

Python 3.11 ou 3.12 recomendado:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

A interface abre com dois bots táticos. **1** demonstração; **2** bancada manual; **3** política PPO; **P** pausa; **R** nova partida; **V** rastreamento; **Tab/clique** seleciona robô azul. WASD/setas controlam o robô selecionado; espaço gira. Com a partida pausada, arraste entidades para montar ensaios. Botão direito + puxar aplica impulso no disco. **[ / ]** percorrem o histórico disponível para replay.

O treino da interface roda em outro processo e salva em `checkpoints/hockey_v2.pt`; use “Carregar pesos” para avaliar. Não existe uma política competente pré-instalada. O antigo `hockey_ppo.pt` é incompatível e não é carregado automaticamente.

## Perfis de arena

```bash
python main.py --profile micro-2024
python main.py --profile micro-large
python main.py --profile tabletop
```

`micro-2024` escolhe 300 × 150 cm dentro da faixa Micro do regulamento RoboCore revisado em 15/01/2024; `micro-large` usa o extremo maior; `tabletop` preserva a mesa didática antiga. Confira revisão, categoria, desenho dos cantos e aceitação de autonomia/câmera com o evento específico. Os perfis distinguem fonte regulamentar de hipóteses de projeto; veja [competição e calibração](docs/COMPETITION_AND_CALIBRATION.md).

## Treinar e avaliar

```bash
python train.py --steps 100000 --envs 4 --seed 42
python train.py --steps 500000 --resume checkpoints/hockey_v2.pt
python train.py --eval-only --resume checkpoints/hockey_v2.best.pt --eval-episodes 30
python evaluate.py --episodes 10 --output runs/benchmark.json
```

O currículo usa finalização, defesa, posições variadas e face-off. `--scenario faceoff` mantém um cenário; `--profile`, `--rollout`, `--episode-steps`, `--save` e `--log` são configuráveis. Os passos informados são transições de ambiente; cada uma gera três amostras de agente. O histórico JSONL inclui KL, fração de clipping, entropia, variância explicada e avaliações em seeds reservadas.

O PPO usa ações Normal→tanh com probabilidades consistentes e GAE separado por ambiente/robô. Truncamentos fazem bootstrap; términos e resets interrompem a recursão. Checkpoints v2 incluem versão da observação, parâmetros, estado do otimizador e configuração física. A carga de um checkpoint incompatível falha explicitamente.

Não interprete aumento de recompensa como vitória. Compare políticas congeladas contra ações zero e bot tático; acompanhe vitórias, gols, finalizações aproximadas, tempo parado e perdas de rastreamento. Treinos curtos servem para verificar o sistema, não para comprovar competência 3v3.

## Da câmera aos robôs

Para inspecionar e gravar o rastreamento antes de treinar uma política:

```bash
python run_robot.py --config configs/robot.local.json --track-only --preview
```

Esse modo registra poses e velocidades sem conectar ao Bluetooth.

```text
Câmera zenital → correção óptica / homografia → ArUco + disco
→ estimador com timestamps → observação v2 compartilhada → PPO
→ supervisor de validade → PWM e BLE → firmware com watchdog
```

As tags ArUco `DICT_4X4_50` usam IDs 0–2 para azul e 3–5 para vermelho; a aresta superior da tag aponta para a frente do robô. A câmera deve ver toda a arena. Velocidade longitudinal e angular vêm das poses; o ator não recebe velocidade verdadeira das rodas nem posse interna do simulador.

Configure a câmera e os quatro pontos de calibração a partir de [configs/robot.example.json](configs/robot.example.json). São exemplos, não calibração válida para sua bancada. Pontos são TL, TR, BR, BL; o mundo usa +y para cima. Para lente calibrada, os pontos devem estar na imagem já sem distorção. Calibre também o plano das tags elevadas ou quantifique o erro de paralaxe.

```bash
python run_robot.py --config configs/robot.local.json --policy checkpoints/hockey_v2.pt --preview
```

Esse comando é **dry-run**, sem conectar ao Bluetooth. Ele verifica a compatibilidade da arena/modelo do checkpoint com a configuração. Estados vencidos ou detecções ausentes bloqueiam a ação. O timestamp de recebimento do frame no computador não é o instante da exposição: meça `capture_delay_seconds` e valide a latência total.

O protocolo BLE v2 deste projeto exige o [firmware próprio](firmware/hockey_v2). Não há alegação de compatibilidade com o firmware original do HockeyBot. Para armar o envio, configure endereços únicos, valide firmware/pinagem em bancada, marque `protocol_verified` e acrescente `--armed`. Não grave o firmware de exemplo sem confirmar a eletrônica. Veja [operação física](docs/ROBOTICS.md).

## Física e CAD

O treino padrão `--opponent mixed` varia a dificuldade do bot, inclui alguns episódios contra ações zero e incorpora versões anteriores da política. Use `--opponent selfplay` para priorizar versões anteriores depois que o primeiro snapshot existir, ou `--opponent heuristic` para manter o bot como adversário. Os snapshots são congelados e a troca de adversário ocorre entre episódios.

Pymunk resolve contatos entre polígonos, disco, traves e paredes. Motores aplicam forças nas rodas; arrasto do patim e atrito de deslizamento são parametrizados. O disco é solto: condução e chutes emergem dos contatos. Gol exige passagem completa entre as traves. A interface conserva tempo e placar entre gols, com partida de cinco minutos e morte súbita em empate.

Frequências padrão: física 240 Hz, controle 30 Hz. Percepção simulada inclui atraso de captura, ruído e perdas; rádio inclui atraso, perdas, ordenação de comandos e watchdog. Os dois usam um gerador por ambiente com seed reproduzível. São modelos configuráveis, sujeitos a identificação experimental.

```bash
python cad/fusion_chassis_generator.py
```

O CAD usa o mesmo contorno da física e do desenho; exporta bases de projeto, não uma montagem comercial certificada. [Detalhes mecânicos](docs/CHASSIS_CAD_PHYSICS_SPEC.md).

## Verificação

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check .
ruff format --check .
```

Os testes cobrem GAE, probabilidades, checkpoint, observações sim/real, tracking, ArUco, puck, gols, colisões, inércia, seeds e protocolo de rádio. A CI executa testes e uma passagem curta do treinamento. Medições físicas, operação dentro do Fusion e desempenho em competição exigem validação na bancada real.
