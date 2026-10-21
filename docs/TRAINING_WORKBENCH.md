# Treinar pela interface

Abra `python main.py`. O padrão é **3v3 self-play**. A janela reúne o campo, o editor do experimento, os controles de execução e os gráficos. O processo de treino é separado para manter a janela responsiva.

## Um robô buscando o disco e marcando

1. Na aba **Treino**, selecione **Solo · buscar e marcar**.
2. Ajuste passos, ambientes paralelos e duração do episódio nos campos. Clique, digite e pressione Enter. Role o painel para os demais controles.
3. Escolha a distância inicial e se o robô começa orientado para o disco ou em uma orientação aleatória. O cenário solo varia as posições a cada episódio.
4. Clique **Iniciar treino**. Existe somente um robô no ambiente físico: os adversários não são obstáculos invisíveis ou robôs parados fora da tela.
5. Use **1×** para observar a coleta em tempo real, **5×** para acelerar ou **Máximo** para usar o desempenho disponível. A exibição acompanha o primeiro ambiente de treino, com até dez quadros amostrados por segundo. Os outros ambientes continuam gerando experiências em paralelo.
6. **Pausar treino** preserva o estado e a coleta. **Salvar e encerrar** encerra de forma cooperativa, atualiza o rollout parcial e salva os pesos. Ao fechar a janela, o programa também solicita salvamento; se o processo não responder, o último checkpoint atômico permanece disponível.

O exercício recompensa aproximação e progresso do disco, e termina com gol, gol contra ou limite de tempo. O disco é livre: o código não o prende ao chassi nem cria chutes artificiais. Marcar com um empurrão ou finalização é permitido; não existe obrigação oculta de contato contínuo.

Para uma primeira habilidade mais simples, escolha **Solo · ir ao disco**. A chegada é definida pela distância entre o centro do robô e o centro do disco, configurável na aba PPO; o padrão é 10 cm. Não exige orientação ou contato. Isso é uma tarefa diferente de dominar o disco ou marcar um gol.

## Exercícios disponíveis

| Preset | Robôs | Objetivo / inicialização |
|---|---|---|
| 3v3 · self-play | 3 azuis + 3 vermelhos | Partida; adversários são snapshots congelados da política desde o primeiro episódio |
| Solo · buscar e marcar | 1 azul | Posições variadas; buscar o disco e marcar no gol da direita |
| Solo · ir ao disco | 1 azul | Posições variadas; sucesso ao entrar na distância de chegada |
| Solo · finalização | 1 azul | Começa perto do disco, avançado no campo |
| 1v1 · defesa | 1 azul + 1 vermelho | Disco lançado na direção do gol azul; adversário tático |
| 3v3 · bot tático | 3 azuis + 3 vermelhos | Partida contra o controlador tático |

Os presets são pontos de partida. Você pode escolher de um a três robôs azuis e de zero a três vermelhos, objetivo, cenário, adversário e arena. Ao selecionar zero vermelhos, o controlador adversário passa a **Sem adversário**. **Parado** conserva robôs vermelhos físicos: use somente quando quiser esses obstáculos.

Os cenários são face-off, solo com posições variadas, finalização, defesa, times em posições variadas e currículo. O currículo alterna finalização, defesa, posições variadas e face-off ao longo do orçamento. Ele não altera a quantidade de robôs ou o objetivo escolhido.

## Ajustes e interpretação

A aba **Treino** contém orçamento, duração, seed, quantidade de robôs, posição inicial e variações de física/visão. A aba **PPO** contém learning rate, gamma, GAE, clipping, entropia, KL alvo, coleta, minibatch, épocas, frequência de snapshots e avaliação. Também permite ajustar os pesos de gol, progresso, aproximação, chegada, tempo e esforço.

Progresso e aproximação usam diferenças de potencial com o mesmo gamma do PPO. Recompensas de simples contato não são adicionadas. Um número alto em uma recompensa personalizada não comprova bom comportamento: inspecione a política e mantenha avaliações reservadas. Desativar variações ajuda na depuração, mas não valida a transferência física.

Passos são **transições de ambiente**, somadas entre os ambientes paralelos. Cada transição gera uma amostra por robô azul: 10 mil passos 3v3 geram 30 mil amostras individuais. O orçamento é arredondado para completar o último lote de ambientes. Os botões de ritmo mudam a velocidade da coleta; os controles da aba **Simular** mudam somente a velocidade da prévia/avaliação independente.

Durante o treino, o formulário fica bloqueado para conservar a identificação do experimento. Pausa, ritmo, visualização e avaliação continuam disponíveis. Para alterar parâmetros, encerre a sessão e inicie outra. A seed identifica uma execução inicial; retomada restaura pesos, otimizador e contadores, mas reinicia episódios, geradores de ambiente e o conjunto de adversários congelados. Ela não reproduz o ponto exato de uma trajetória interrompida.

Os gráficos mostram **retorno e sucesso na coleta**, em uma janela de até vinte episódios concluídos. Esses episódios usam ações exploratórias e adversários que podem mudar. Na aba PPO aparecem KL, clipping, entropia e variância explicada. A avaliação periódica usa ações determinísticas, seeds reservadas e o mesmo objetivo/quantidade de robôs, contra bot tático fixo quando há adversários, ou sem adversário no solo. O adversário parado é mantido quando essa é a escolha do experimento.

## Ver o que está aprendendo

- **Ver treino ao vivo:** mostra posições e ações do primeiro ambiente que efetivamente preenche o rollout. Durante uma atualização ou avaliação reservada, a coleta fica parada e o último quadro permanece visível.
- **Avaliar últimos pesos:** cria uma simulação independente com uma cópia congelada do último checkpoint. O treino continua. Recarregue para inspecionar uma atualização mais recente.
- **Pré-visualizar exercício:** usa o bot tático para conferir a montagem e o cenário, sem aprendizado. O cabeçalho identifica essa prévia.
- **Simular:** demonstração, controle manual, pausa, velocidade, tracking e replay. Pausado, arraste robôs ou disco; botão direito aplica impulso. Essas intervenções alteram a prévia, não o ambiente de coleta.

O cabeçalho distingue coleta, avaliação, prévia e replay. A janela não usa uma partida tática independente para representar falsamente o que o PPO está fazendo.

## Salvar, retomar e transferir

Cada início cria uma pasta em `runs/experiments/<data-id>/` com:

| Arquivo | Conteúdo |
|---|---|
| `config.json` | Parâmetros completos do experimento |
| `latest.pt` | Pesos, otimizador, contadores e configuração física; salvo a cada atualização e ao encerrar |
| `latest.best.pt` | Melhor taxa de sucesso observada nas avaliações reservadas; pode não existir antes da primeira avaliação |
| `metrics.jsonl` / `episodes.jsonl` | Métricas das atualizações e resultados de episódios |
| `live.json` / `status.json` | Último quadro e estado da execução |
| `control.json` / `output.log` | Solicitações de pausa/ritmo/parada e diagnóstico do processo |

A aba **Sessões** permite restaurar uma configuração, avaliar, preparar uma retomada ou preparar uma transferência. Retomar conserva o estado do otimizador e exige compatibilidade de tarefa, cenário, adversário, física e hiperparâmetros. O orçamento da nova sessão é adicional ao histórico dos pesos. Transferir importa a rede, inicia um novo otimizador e zera os contadores do experimento, permitindo mudar de solo para 3v3. O vetor compartilhado continua com 63 entradas por robô; entidades ausentes ficam sem confiança, em vez de inventar posições.

Também é possível arrastar um `.pt` ou um JSON de experimento para a janela. Para avaliar um arquivo selecionado no exercício atual, use o botão correspondente na aba PPO. **Salvar configuração** cria `runs/training-config.json`, que pode ser carregado na aba Sessões. Os seis exemplos reutilizáveis estão em `configs/training/`.

## Execução opcional por terminal

```bash
python main.py --preset solo_goal
python main.py --training-config configs/training/solo_goal.json
python train.py --config configs/training/solo_goal.json --steps 100000
python train.py --config runs/experiments/SESSAO/config.json --resume runs/experiments/SESSAO/latest.pt
python train.py --config configs/training/selfplay.json --init-from runs/experiments/SESSAO_SOLO/latest.pt
python evaluate.py --policy runs/experiments/SESSAO/latest.pt --episodes 20
```

O benchmark por terminal lê a tarefa e a quantidade de robôs do checkpoint. Pesos compatíveis em dimensão não implicam uma política competente em outra tarefa. Um treino solo não ensina cooperação ou defesa 3v3; avalie essas habilidades depois da transferência.

A interface preserva o contrato de observação por câmera zenital. Os limites do modelo do HockeyBot e a integração Bluetooth original pendente continuam descritos em [HOCKEYBOT_STOCK.md](HOCKEYBOT_STOCK.md).
