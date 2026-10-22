# 1v1 self-play: treino noturno e teste contra uma pessoa

O GIF da POC foi um exemplo selecionado de condução aprendida em uma tarefa facilitada. O treino curto acumulou 65.536 transições e 32 atualizações; ainda não estabeleceu competência para uma partida contra uma pessoa. Um treino noturno amplia a experiência, mas a qualidade precisa ser medida por avaliações e jogos.

## POC preservada

Os pesos do GIF foram copiados para `checkpoints/poc-solo-20261021.pt`. O arquivo `checkpoints/poc-solo-20261021.manifest.json` registra origem, configuração, resultados de validação e SHA-256. A cópia foi conferida antes do treino 1v1. O novo treino carrega a rede por transferência, inicia um novo otimizador e salva em outra pasta. A POC continua disponível para demonstração.

## Configuração do 1v1

O preset **1v1 · self-play** (`duel_selfplay`) usa um azul que aprende e um vermelho controlado por uma versão congelada da política. Cada ambiente escolhe adversários entre até doze snapshots recentes e a política inicial preservada, com probabilidade de 20% para a inicial. O conjunto e a âncora são incluídos nos checkpoints para retomada. Snapshots são atualizados a cada dez atualizações PPO.

O cenário `duel` mistura face-off (35%), posições e orientações variadas (35%), ataque perto do gol (15%) e defesa da situação espelhada (15%). Oponentes, disco e colisões continuam reais na física. A rede usa o mesmo contrato de observação por câmera zenital. Não há controlador tático oculto na coleta com `opponent=selfplay`.

| Parâmetro inicial | Valor |
|---|---:|
| Arena | `micro-practice`, 300 × 150 cm, gols recuados |
| Ambientes | 8 trajetórias, coleta vetorizada |
| Episódio de treino | Até 60 s ou o primeiro gol |
| Rollout por ambiente | 256 passos |
| Minibatch / épocas | 512 / 6 |
| Learning rate | 0,0002 |
| Gamma / GAE | 0,997 / 0,95 |
| Entropia | 0,005 |
| Snapshots | A cada 10 updates; pool de até 12 |
| Avaliação | A cada 100 updates, 6 episódios por adversário |

São parâmetros de partida, sem garantia de vitória contra uma pessoa. A física usa CPU e os ambientes são percorridos pelo worker; oito ambientes não significam oito processos de física.

## Avaliação que não muda junto com a liga

A suíte reservada usa ações determinísticas contra o bot tático fixo e contra a política inicial congelada. Cada adversário enfrenta os mesmos seis seeds reservados, alternando face-off e posições variadas, com episódios de até sessenta segundos. As trajetórias de avaliação não entram no PPO.

`success_rate` é a fração de episódios vencidos; também são registrados derrotas e limites de tempo. `point_score` conta vitória como 1 e limite de tempo como 0,5. No treino original de 21/10, o melhor checkpoint foi escolhido por esse índice. Após a análise de 22/10, novos treinos selecionam `latest.best.pt` por **`active_success_rate`**, exigindo participação do azul (ou chegada no objetivo `reach`) e avaliação completa. Um gol contra do adversário sem toque azul continua valendo no placar, mas não promove o modelo como habilidade aprendida. Checkpoints antigos continuam carregáveis; o limiar do seletor anterior é reiniciado na nova sessão porque as métricas diferem. O resultado de doze duelos curtos é uma medida periódica, não uma estatística de desempenho contra humanos ou em partidas completas de cinco minutos.

O preset `duel_selfplay` passa a usar `stagnation_patience=200`: encerra cooperativamente após duzentos episódios consecutivos sem sucesso ativo. Configurações JSON antigas mantêm o padrão desativado; use `--stagnation-patience 200` explicitamente ao retomá-las. Para recuperar fundamentos antes de nova disputa, veja [o protocolo solo](FOUNDATION_TRAINING.md).

`latest.pt` é salvo após cada atualização e ao encerrar. Os pesos da POC permanecem fora dessa pasta. Para uma avaliação maior após o treino:

```powershell
.\.venv\Scripts\python.exe train.py --eval-only --resume runs/experiments/duel-overnight-20261021/latest.pt --eval-episodes 20 --output json
```

## Executar durante a noite

`configs/training/duel_selfplay.json` é o preset reutilizável. `configs/training/duel_overnight.json` aponta para a cópia da POC e amplia o limite de passos; o encerramento noturno é definido por tempo real, separado da configuração de aprendizado.

```powershell
# Exemplo com duração de oito horas
.\.venv\Scripts\python.exe train.py --config configs/training/duel_overnight.json --run-dir runs/experiments/duel-overnight-20261021 --max-hours 8 --keep-awake --no-live --output json

# Alternativa: horário absoluto, com fuso explícito
.\.venv\Scripts\python.exe train.py --config configs/training/duel_overnight.json --until 2026-10-22T07:00:00-03:00 --keep-awake --no-live
```

As duas opções de limite são alternativas. O worker solicita parada cooperativa no horário indicado, termina a atualização parcial e salva. `status.json` registra `stop_reason=time_limit`. `runtime.json` registra PID, limite e opções de execução. O comando de horário absoluto acima é específico da noite de 21 para 22/10/2026; ajuste-o em outras datas.

No Windows, `--keep-awake` inibe a suspensão automática enquanto o processo treina e libera essa solicitação ao sair. A tela pode apagar. Desligamento, suspensão manual e fechamento da tampa podem interromper o processo; o último checkpoint completo fica disponível para retomada. A configuração global de energia não é alterada.

Para consultar e parar uma sessão iniciada em outro terminal:

```powershell
.\.venv\Scripts\python.exe train.py --status runs/experiments/duel-overnight-20261021
.\.venv\Scripts\python.exe train.py --pause runs/experiments/duel-overnight-20261021
.\.venv\Scripts\python.exe train.py --continue runs/experiments/duel-overnight-20261021
.\.venv\Scripts\python.exe train.py --stop runs/experiments/duel-overnight-20261021
```

Uma retomada usa uma pasta nova e recupera pesos, otimizador, política inicial e pool de snapshots; episódios e geradores de ambiente reiniciam:

```powershell
.\.venv\Scripts\python.exe train.py --resume runs/experiments/duel-overnight-20261021/latest.pt --steps 100000000 --max-hours 8 --keep-awake --no-live
```

## Você contra a IA na simulação

Após existirem pesos atualizados, abra uma partida independente, sem interromper o worker:

```powershell
.\.venv\Scripts\python.exe main.py --training-config runs/experiments/duel-overnight-20261021/config.json --policy runs/experiments/duel-overnight-20261021/latest.pt --human
```

Você controla o **vermelho** com WASD ou setas; espaço gira, P pausa e R reinicia. A IA controla o **azul** com pesos congelados. A tecla **4** ou o botão **Humano × PPO**, na aba Simular, seleciona esse modo. Ele exige um robô de cada lado. Placar e relógio são preservados entre gols, seguindo o tempo da arena e a morte súbita já existentes. Para carregar pesos mais novos, reabra a partida ou recarregue a avaliação.

O teste em robôs físicos continua dependendo da calibração de percepção/dinâmica e do transporte de comandos; a integração Bluetooth original do HockeyBot permanece pendente. Treinar em self-play não comprova, por si só, desempenho contra uma pessoa no equipamento físico.
