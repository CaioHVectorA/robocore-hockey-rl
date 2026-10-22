# Prompt para o Antigravity

**Atualização de 22/10/2026:** a sessão abaixo terminou estagnada (7 vitórias em 5.104 episódios de coleta, nenhuma após o episódio 344). As instruções de retomada competitiva deste documento são históricas; não repetir outra noite automaticamente. Para o próximo experimento, seguir [FOUNDATION_TRAINING.md](FOUNDATION_TRAINING.md), com `solo_foundation`, participação, validação de etapa e parada por estagnação. Os pesos antigos devem permanecer preservados para comparação.

Continue o treinamento 1v1 de hockey neste projeto Windows:
`C:\Users\caihe\Documents\develop\robocore-hockey-rl`.
Use PowerShell e o Python da `.venv`. Execute o treino e acompanhe os resultados com atualizações reais no chat. O objetivo é um robô autônomo localizar o disco, disputar e marcar gol; depois quero jogar contra a política na simulação e usar isso como POC para transferência ao robô físico.

## Estado confirmado da entrega

- O worker anterior, PID 76788, foi encerrado cooperativamente em 21/10/2026 às 20:22, horário de Brasília. `status.json` está `stopped`, com `stop_reason=requested`; o processo já saiu.
- Checkpoint de retomada: `runs/experiments/duel-overnight-20261021/latest.pt`. Foi carregado e validado depois da parada: 1.221.152 transições, 597 atualizações PPO e pool de 12 adversários. Inclui pesos, otimizador, política inicial e pool. Episódios e geradores de ambiente reiniciam na retomada.
- Para uma cópia nova do GitHub, use `checkpoints/duel-selfplay-20261021.pt`: é uma cópia idêntica desse checkpoint final. O melhor anterior está em `checkpoints/duel-selfplay-20261021.best.pt`; os hashes e a avaliação estão no manifesto ao lado. Substitua o caminho de origem nos comandos abaixo pelo caminho versionado quando `runs/` não estiver disponível. O GIF versionado está em `docs/assets/ppo-conducao-gol.gif`.
- `runs/experiments/duel-overnight-20261021/latest.best.pt` também existe. Continue de `latest.pt`; preserve ambos. O melhor checkpoint é selecionado pela avaliação, não por ser o mais recente.
- A POC solo está preservada em `checkpoints/poc-solo-20261021.pt`. NÃO sobrescreva. SHA-256: `9d39ef524e10b05e44f96ea801d9f7e4f02b74e1e6335d514fb12ff45d593f6a`. Seu GIF é `runs/gifs/ppo-conducao-gol.gif`.
- A última avaliação completa registrada na parada teve 12 episódios: 0 vitórias, 2 derrotas e 10 limites de tempo. Portanto, não considere o 1v1 resolvido. O melhor `point_score` histórico salvo é 0,5416667; esse número NÃO é taxa de vitória.
- Há alterações locais importantes ainda sem commit. Preserve-as. Leia `docs/OVERNIGHT_1V1.md`, `docs/CLI.md` e o código antes de alterar o treinamento. A suíte anterior passou com 126 testes e 2 skips; não precisa rodar tudo novamente apenas para iniciar o worker.

## Retomar e acompanhar

Primeiro confirme que não há outro worker treinando essa sessão, consultando `runtime.json`, `status.json` e a linha de comando dos processos. Não encerre processos Python alheios. Não inicie duas cópias.

Use uma NOVA pasta de sessão, sem reaproveitar a pasta encerrada nem copiar seu `control.json`, que contém `stop=true`. O horário antigo de 07:00 foi cancelado. Use inicialmente um orçamento de até oito horas a partir da nova execução, sujeito a interrupção minha; não há necessidade de esperar até amanhã.

```powershell
Set-Location -LiteralPath 'C:\Users\caihe\Documents\develop\robocore-hockey-rl'
$hockeyRunDir = 'runs/experiments/duel-antigravity-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
.\.venv\Scripts\python.exe -u train.py --resume runs/experiments/duel-overnight-20261021/latest.pt --run-dir $hockeyRunDir --steps 100000000 --max-hours 8 --keep-awake --no-live --output json
```

Esse comando é bloqueante. Para continuar acompanhando pelo agente, use execução em segundo plano suportada pelo seu ambiente ou `Start-Process -WindowStyle Hidden`, com diretório correto e stdout/stderr em arquivos separados. Registre o caminho absoluto da nova sessão e o PID real que aparecer em `runtime.json`. O launcher da `.venv` pode ter PID diferente do worker. Não dependa de um terminal temporário que mate o processo ao fechar. Os 100 milhões são um teto de transições; o limite real dessa execução é de oito horas.

Use `--resume` para recuperar o estado do treinamento. Não reinicie com `configs/training/duel_overnight.json` sozinho: ele está configurado para transferência da POC solo e perderia a continuidade do 1v1. Não use `--init-from` nesta retomada.

Confirme no início que o contador de updates continuou acima de 597, a liga foi recuperada, as métricas avançam e um novo `latest.pt` foi salvo. O `session_transitions` da nova pasta recomeça em zero; os contadores acumulados do treinador continuam.

Envie uma atualização após confirmar a retomada, depois a cada dez minutos e imediatamente em erro, avaliação relevante ou encerramento. Use recursos reais de monitoramento/agendamento disponíveis no seu ambiente. Se você não puder emitir mensagens futuras autonomamente, explique essa limitação e mantenha logs verificáveis; não prometa notificações que não conseguirá enviar.

Em cada atualização, informe tempo decorrido, passos da sessão/acumulados, updates, taxa de coleta medida, checkpoint salvo e resultados da avaliação mais recente, indicando a qual update ela pertence. Separe vitórias/derrotas/limites de tempo contra cada adversário. Não confunda o sucesso dos últimos 20 episódios de treino com a avaliação reservada. Não publique atualização sem ler dados novos.

Leia `status.json`, `metrics.jsonl`, `episodes.jsonl` e os logs. `--no-live` desativa apenas a publicação de poses para a UI, mantendo física e salvamentos. Não altere código/recompensas/hiperparâmetros silenciosamente durante um worker ativo. Se houver estagnação, apresente evidências e uma proposta; uma mudança de tarefa ou recompensa pode exigir transferência para outra experiência, em vez de `resume`.

## Configuração e métricas existentes

- Cenário `duel`, objetivo `score`, um azul aprendendo e um vermelho em self-play; arena `micro-practice`, 300 x 150 cm, gols em recuos nas extremidades curtas. A profundidade de 26 cm é estimada, não uma medida confirmada da pista física.
- Oito ambientes coletados pelo worker em CPU, não oito processos físicos. Rollout 256, batch 512, seis épocas, episódio até 60 s ou primeiro gol; learning rate 0,0002, gamma 0,997, GAE 0,95, entropia 0,005.
- Pool de até 12 snapshots, atualizado a cada dez updates, mais a política inicial congelada com probabilidade de 20%.
- Avaliação a cada 100 updates, seis episódios por adversário, contra bot tático fixo e política inicial, com seeds reservados. Não usa as trajetórias de avaliação para aprender.
- `point_score = (vitórias + 0,5 * limites de tempo) / episódios`. `latest.best.pt` depende de melhoria nesse critério e avaliação completa; muitos empates por tempo podem mascarar falta de gols. Compare também vitórias e gols, não só pontuação/reward.
- `latest.pt` é sobrescrito atomicamente após cada update e ao sair. Os snapshots da liga estão dentro do checkpoint; não são um arquivo permanente por update. Para histórico adicional, copie o `latest.pt` para nomes com timestamp aproximadamente a cada 30 minutos usando leitura consistente: preferencialmente bytes de um único handle, fechando-o prontamente e depois validando a cópia com o loader. Se houver corrida de substituição no Windows, tente novamente sem interromper o treino. Preserve o melhor checkpoint anterior.

## Controle e entrega

Substitua `<NOVA_PASTA>` pelo caminho real, sem os sinais de menor/maior:

```powershell
.\.venv\Scripts\python.exe train.py --status <NOVA_PASTA>
.\.venv\Scripts\python.exe train.py --stop <NOVA_PASTA>
```

Ao receber pedido de parada ou atingir o orçamento, aguarde `stopped`/`completed`, confirme saída do worker e carregue o checkpoint final. Use parada cooperativa, evitando matar o processo no meio do salvamento. Não reinicie automaticamente após um pedido meu de parada.

Para uma avaliação maior, preferencialmente após parar ou usando uma cópia consistente:

```powershell
.\.venv\Scripts\python.exe train.py --eval-only --resume <NOVA_PASTA>/latest.pt --eval-episodes 20 --output json
```

Para eu jogar contra ele:

```powershell
.\.venv\Scripts\python.exe main.py --training-config <NOVA_PASTA>/config.json --policy <NOVA_PASTA>/latest.pt --human
```

Humano controla o vermelho com WASD/setas; espaço gira. PPO controla o azul. Os pesos dessa partida ficam congelados; reabra para carregar pesos novos.

Na entrega, informe os caminhos dos checkpoints, configuração, tempo, transições e avaliações comparáveis. Se gerar um GIF, use ações reais da política e identifique se foi um episódio selecionado. Treinar por horas em self-play não comprova inteligência contra humanos nem transferência ao robô físico: percepção, dinâmica e transporte de comandos ainda precisam de validação/calibração.
