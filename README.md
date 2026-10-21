# Hockey Lab — simulação, PPO e visão zenital

Simulador 2D de hockey 3v3 com contatos de corpos rígidos, controle diferencial e uma arquitetura compartilhada entre percepção simulada e câmera real. O projeto é uma plataforma de desenvolvimento: parâmetros de hardware ainda precisam ser medidos e uma política precisa ser treinada e avaliada antes de uso nos robôs.

O hardware de referência é o **HockeyBot pronto da RoboCore, controlado pelo aplicativo Bluetooth RoboCore Joystick**. Os três papéis usam o mesmo chassi, com recorte frontal curvo e rodas expostas na traseira. Massa aproximada de 360 g, pneus de 60 × 20 mm e dados do motor vêm do fabricante; dimensões do chassi e dinâmica ainda são estimadas. [Fontes, hipóteses e limites de fidelidade](docs/HOCKEYBOT_STOCK.md).

## Instalar e jogar

Python 3.11 ou 3.12 recomendado:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

A interface abre na **bancada de treino**, com o padrão **3v3 self-play**. Escolha um exercício, ajuste os campos e clique **Iniciar treino**. O campo mostra o primeiro ambiente que está realmente gerando experiências do PPO. Há pausa, ritmo 1×/5×/máximo, encerramento com salvamento, gráficos e histórico de sessões.

Para treinar um robô sozinho a buscar o disco e levar ao gol, selecione **Solo · buscar e marcar**. **Solo · ir ao disco** ensina a aproximação; finalização e defesa também têm presets. Quantidade de robôs, cenário, duração, seed, recompensas e parâmetros do PPO são editáveis. Os adversários do treino solo não existem na física. Veja o [guia da bancada](docs/TRAINING_WORKBENCH.md).

**Ver treino ao vivo** acompanha a coleta exploratória. **Avaliar últimos pesos** abre uma simulação independente com pesos congelados. **Pré-visualizar exercício** usa o bot tático e é identificado como prévia. A aba **Simular** preserva controle manual, pausa, arraste, impulso, tracking e replay. **1/2/3** alternam controlador; **P** pausa; **V** tracking; **Tab** seleciona; WASD/setas e espaço controlam o robô na bancada manual.

Cada treino salva configuração, pesos, métricas e logs em `runs/experiments/`. A aba **Sessões** permite avaliar, retomar ou transferir pesos entre exercícios. Não existe uma política competente pré-instalada. O antigo `hockey_ppo.pt` é incompatível e não é carregado automaticamente.

## Perfis de arena

```bash
python main.py --profile micro-2024
python main.py --profile micro-large
python main.py --profile tabletop
```

`micro-2024` escolhe 300 × 150 cm dentro da faixa Micro do regulamento RoboCore revisado em 15/01/2024; `micro-large` usa o extremo maior; `tabletop` preserva a mesa didática antiga. Confira revisão, categoria, desenho dos cantos e aceitação de autonomia/câmera com o evento específico. Os perfis distinguem fonte regulamentar de hipóteses de projeto; veja [competição e calibração](docs/COMPETITION_AND_CALIBRATION.md).

A categoria comercial HockeyBot não foi presumida equivalente à Micro. A referência do robô e o perfil de arena são independentes; o perfil padrão não comprova as dimensões da arena de um evento HockeyBot.

## Treinar e avaliar

```bash
python main.py --preset solo_goal
python train.py --config configs/training/selfplay.json --steps 100000
python train.py --config configs/training/solo_goal.json --steps 100000
python train.py --config runs/experiments/SESSAO/config.json --resume runs/experiments/SESSAO/latest.pt
python evaluate.py --policy runs/experiments/SESSAO/latest.pt --episodes 20
```

O currículo opcional usa finalização, defesa, posições variadas e face-off. Os passos são transições de ambiente; cada uma gera uma amostra por robô azul. A interface mostra retorno e sucesso da coleta; o histórico inclui KL, clipping, entropia, variância explicada e avaliações em seeds reservadas. As avaliações preservam o exercício e a quantidade de robôs. Transferência carrega a rede em um novo otimizador; retomada exige compatibilidade do experimento.

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

O exemplo usa `transport: robocore-stock`. O protocolo Bluetooth original ainda não foi implementado: visão e dry-run funcionam, mas `--armed` é rejeitado. A ponte BLE v2 deste projeto exige um [adaptador com firmware próprio](firmware/hockey_v2) e eletrônica compatível. Para esse adaptador, selecione `transport: hockey-v2-custom`, configure endereços únicos, valide firmware/pinagem em bancada, marque `protocol_verified` e acrescente `--armed`. Não grave o firmware de exemplo sem confirmar a eletrônica. Veja [operação física](docs/ROBOTICS.md).

## Física e CAD

O treino padrão `--opponent selfplay` usa versões congeladas da própria política desde o primeiro episódio, com exploração nos dois times. `--opponent mixed` varia bot, adversários parados e snapshots; `--opponent heuristic` mantém o bot tático. A troca de adversário ocorre entre episódios. No solo, o controlador é `none` e não há robôs vermelhos.

Pymunk resolve contatos entre polígonos, disco, traves e paredes. Motores aplicam forças nas rodas; arrasto do patim e atrito de deslizamento são parametrizados. O disco é solto: condução e chutes emergem dos contatos. Gol exige passagem completa entre as traves. A interface conserva tempo e placar entre gols, com partida de cinco minutos e morte súbita em empate.

Frequências padrão: física 240 Hz, controle 30 Hz. Percepção simulada inclui atraso de captura, ruído e perdas; rádio inclui atraso, perdas, ordenação de comandos e watchdog. Os dois usam um gerador por ambiente com seed reproduzível. São modelos configuráveis, sujeitos a identificação experimental.

```bash
python cad/fusion_chassis_generator.py
python -m ui.robot_renderer --output runs/hockeybot-reference.png
```

O CAD usa o mesmo contorno da física e do desenho; exporta placas planas de referência estimadas por foto, não uma montagem comercial certificada. O segundo comando gera uma vista ampliada do robô. [Detalhes mecânicos](docs/CHASSIS_CAD_PHYSICS_SPEC.md). Pesos anteriores à revisão física do HockeyBot precisam de novo treinamento.

## Verificação

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check .
ruff format --check .
```

Os testes cobrem GAE, probabilidades, checkpoint, observações sim/real, tracking, ArUco, puck, gols, colisões, inércia, seeds e protocolo de rádio. A CI executa testes e uma passagem curta do treinamento. Medições físicas, operação dentro do Fusion e desempenho em competição exigem validação na bancada real.
