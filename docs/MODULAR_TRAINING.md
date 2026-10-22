# PPO modular: aprovar habilidades antes de avançar

O coordenador `train_modular.py` executa os módulos de `configs/training/modular.json` em sequência, transferindo a política aprovada e mantendo a mesma observação de 63 valores e duas ações de roda. Não existe promoção por tempo gasto, quantidade de passos ou recompensa de coleta.

## Sequência e aprovação

1. Aproximar alinhado, em 1v0.
2. Aproximar com erro inicial de até 45 graus.
3. Localizar e girar com erro inicial de até 120 graus.
4. Chegar de qualquer direção/orientação.
5. Aproximar com velocidade relativa ao disco limitada a 30 cm/s na chegada.
6. Finalizar perto do gol.
7. Conduzir até o gol.
8. Buscar de qualquer direção, conduzir e marcar.
9. Jogar 1v1 contra adversário parado.
10. Jogar 1v1 contra bot simples (força 0,35).
11. Self-play, com avaliação contra bot de força 0,85 e política inicial congelada do módulo.

Um módulo precisa de **80% de sucesso ativo em cinquenta episódios de validação** e depois **80% em outros cinquenta de confirmação**, com seeds distintas. No self-play, cada adversário fixo precisa passar individualmente; uma média agregada não esconde um adversário fraco. O coordenador verifica também os módulos anteriores marcados `retain`, com vinte episódios cada, exigindo 80%. São preservadas as tarefas anteriores de marcar gol; chegada como objetivo isolado não é exigida novamente após aprender a continuar conduzindo.

Essas metas definem uma aprovação operacional, não uma garantia de ótimo global ou de competência no robô físico. As faixas de seeds são validação de desenvolvimento, pois são revisitadas a cada coleta. Avaliações finais externas e jogos humanos continuam necessários. As seeds de treino ficam em faixas distintas das provas.

As primeiras etapas de chegada têm recompensa de gol zero. Marcar sem cumprir a aproximação não aprova uma tarefa de chegada. A etapa de velocidade exige distância de até 12 cm e velocidade **relativa** de até 30 cm/s; não é uma demonstração de rastreamento de qualquer velocidade de referência. Há penalidade de excesso de velocidade relativa a até 30 cm do disco. As tarefas de marcar continuam contando gols reais com participação do azul.

## Coleta, proteção e transferência

Cada coleta tem até 65.536 transições; após ela são salvos pesos, avaliação e um GIF. Um módulo tem teto de 1.048.576 transições e para após oito verificações sem melhoria relevante no indicador de progresso (sucesso, contatos, distância e inatividade), ou quatrocentos episódios consecutivos sem sucesso ativo. Ao atingir um limite sem aprovação, o status vira `needs_attention`; o próximo módulo não inicia. O limite global padrão é quatro horas por execução.

Na mudança de módulo, o ator é transferido, o otimizador é novo e o plano reinicia crítico e exploração. Entre coletas do mesmo módulo, pesos, crítico, otimizador e liga são retomados. Estados dos ambientes reiniciam com novas seeds. O plano usa o modelo experimental preservado de 22/10 como origem padrão; os checkpoints anteriores permanecem intactos.

Os flags `--reset-critic` e `--reset-exploration` também existem na CLI comum e só têm efeito em transferência. `--heading-spread` é o desvio angular de inicialização no solo, em radianos; `--reach-speed-limit` configura o limite relativo. Padrões antigos continuam compatíveis: desvio de pi, limite de velocidade desativado, sem reinicializações automáticas.

## Executar e controlar

```powershell
.\.venv\Scripts\python.exe -u train_modular.py --run-dir runs/modular/MEU_TREINO --max-hours 4 --keep-awake

.\.venv\Scripts\python.exe train_modular.py --status runs/modular/MEU_TREINO
.\.venv\Scripts\python.exe train_modular.py --pause runs/modular/MEU_TREINO
.\.venv\Scripts\python.exe train_modular.py --continue runs/modular/MEU_TREINO
.\.venv\Scripts\python.exe train_modular.py --stop runs/modular/MEU_TREINO

# Após interrupção ou fim do orçamento global:
.\.venv\Scripts\python.exe -u train_modular.py --resume runs/modular/MEU_TREINO --max-hours 4 --keep-awake
```

Use nova pasta ao iniciar uma experiência diferente. Para uma execução em segundo plano no Windows, use `Start-Process -WindowStyle Hidden`, diretório correto e arquivos separados para stdout/stderr. O PID real fica em `runtime.json`. `--keep-awake` inibe a suspensão automática enquanto o processo roda; tela apagada é permitida. Desligamento ou suspensão manual ainda interrompem o trabalho.

Parada é cooperativa e salva o rollout parcial. A retomada recupera também um checkpoint de uma coleta que foi interrompida antes de o coordenador registrar seu encerramento. Não há duas instâncias permitidas pelo fluxo normal de uso; confira o PID antes de retomar uma sessão que pareça abandonada. Um estágio em `needs_attention` conserva os limites atingidos: retomar sem revisar o experimento não remove a reprovação.

## GIFs, modelos e resultados

Na raiz da execução:

- `index.html` e `INDEX.md`: galeria dos comportamentos já registrados.
- `gifs/000-current-*.gif`: comportamento inicial de condução e busca, antes de treinar.
- `gifs/NN-modulo-before.gif`: antes de cada módulo.
- `gifs/NN-modulo-NNN.gif`: depois de cada coleta.
- `gifs/NN-modulo-approved.gif`: comportamento dos pesos aprovados.
- `status.json`, `events.jsonl`, `state.json`, `plan.json`: estado, decisões e plano fixado.

Cada GIF usa a seed fixa 777 e ações determinísticas reais do PPO. A seed não é trocada para buscar um gol bonito. O JSON ao lado contém configuração, hash dos pesos, seed, resultado, duração e participação. Clipes têm até vinte segundos simulados e pausa final de 800 ms; `clip_limit` significa fim do clipe, não vitória nem timeout da avaliação. No self-play, o GIF mostra o adversário inicial congelado, identificado no JSON.

Em cada pasta de módulo, `approved.pt` só é criado após as provas e não é sobrescrito. Seu `approval.json` guarda provas, regressões e SHA-256. `best.pt` é o melhor candidato pelo indicador de progresso; `latest.pt` é o fim da última coleta. Eles podem existir sem aprovação. As pastas `train-NNN` preservam cada coleta completa e seus logs. Se um módulo já passar com os pesos de entrada, o checkpoint aprovado é a cópia exata desses pesos, sem simular atualizações que não ocorreram.

Para exportação em outra máquina, instale as dependências da simulação e `requirements-recording.txt`. O renderer usa SDL dummy, sem abrir uma janela. Não há atuação em hardware físico neste comando.
