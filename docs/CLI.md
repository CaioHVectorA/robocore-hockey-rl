# Treino pelo terminal, sem janela

`train.py` executa o mesmo PPO e a mesma física da bancada sem importar Pygame, abrir janelas ou usar um display. Também existe a entrada `python main.py --headless`, com os mesmos argumentos. As configurações JSON e os checkpoints são compartilhados com a UI.

## Começar com o exercício solo

Na pasta do projeto, com as dependências instaladas:

```powershell
.\.venv\Scripts\python.exe train.py --preset solo_goal --steps 100000 --no-live
```

Ou, com o ambiente virtual ativado:

```bash
python main.py --headless --preset solo_goal --steps 100000 --no-live
python train.py --list-presets
python train.py --help
```

O preset `solo_goal` cria um robô azul, nenhum adversário e o objetivo de buscar o disco e marcar no gol da direita. `solo_reach` encerra ao chegar perto do disco. O terminal mostra percentual, passos/s, tempo, fases do PPO, episódios, retorno, sucesso e resultados da avaliação reservada. Retorno e sucesso da coleta usam episódios exploratórios, não comprovam desempenho da política congelada.

`solo_curriculum` oferece [treino solo progressivo por sucesso](ARENA_AND_SOLO_TRAINING.md), com os mesmos pesos em três etapas. Use `--curriculum-success-threshold`, `--curriculum-min-episodes` e `--curriculum-window` para configurar a promoção. O perfil inicial é `micro-practice`, com a pista e os gols recuados da referência de bancada. Os eventos de progresso incluem a etapa e a taxa de sucesso; as métricas incluem o retorno separado por componente de recompensa.

`--no-live` elimina apenas a gravação de poses em `live.json`. Física, percepção simulada, controle, recompensas, PPO, métricas e checkpoints continuam funcionando. Sem essa opção, o worker também publica as poses, ainda sem abrir uma janela. O ritmo padrão é máximo (`--speed 0`); um JSON carregado ou checkpoint retomado pode ter outro ritmo. Use `--speed 0` para sobrescrevê-lo.

## Recompensas e hiperparâmetros

As opções da CLI sobrescrevem o preset ou JSON selecionado. `--config` e `--preset` são alternativas para a configuração inicial. As recompensas não especificadas conservam seus valores anteriores.

```bash
python train.py --preset solo_goal --steps 200000 --envs 4 --learning-rate 0.0003 --gamma 0.995 --entropy-coef 0.002 --rollout 256 --batch-size 256 --epochs 6 --reward goal=25 --reward approach=1.2 --reward progress=3 --reward time_penalty=0.006 --no-live
```

`--reward NOME=VALOR` pode ser repetido. Aceita ponto ou vírgula decimal; os valores precisam ser finitos e não negativos:

| Nome | Efeito |
|---|---|
| `goal` | Bônus por gol e penalidade por gol contra |
| `progress` | Peso do progresso do disco até o gol, no objetivo de marcar |
| `approach` | Peso da aproximação robô–disco |
| `time_penalty` | Custo subtraído por passo de controle |
| `energy_penalty` | Custo proporcional à média das ações ao quadrado |
| `reach_bonus` | Bônus de sucesso no objetivo de ir ao disco |
| `reach_distance` | Distância entre centros para concluir a chegada, de 4 a 30 cm |

Progresso e aproximação são diferenças de potencial, não bônus fixos por toque. Bônus e distância de chegada não concluem o exercício de buscar e marcar.

Também estão disponíveis `--gae-lambda`, `--clip-eps`, `--target-kl`, `--snapshot-every`, `--eval-every` e `--eval-episodes`. O cenário aceita `--episode-seconds`, `--spawn-distance`, `--no-random-heading` e `--no-domain-randomization`. A frequência física e a taxa de controle não mudam com o ritmo da coleta. `--episode-steps` continua disponível como alternativa à duração em segundos.

Confira e exporte a configuração resolvida sem iniciar um treino:

```bash
python train.py --preset solo_goal --reward approach=1.2 --print-config --save-config configs/solo.local.json
python train.py --config configs/solo.local.json --steps 100000 --no-live
```

`--dry-run` é um alias de `--print-config`: valida os parâmetros e mostra o JSON, sem criar uma sessão. Para editar todos os valores num arquivo, parta dessa exportação ou dos exemplos em `configs/training/`.

## Salvar e continuar

```bash
python train.py --preset solo_goal --steps 100000 --run-dir runs/experiments/solo-cli --no-live
python train.py --resume runs/experiments/solo-cli/latest.pt --steps 50000 --no-live
```

Cada início cria uma sessão nova. `--run-dir` permite nomear sua pasta e recusa reutilizar uma sessão existente. `--resume` recupera a configuração do checkpoint automaticamente, preserva pesos e otimizador e acrescenta o orçamento de passos informado. Episódios e geradores de ambiente reiniciam; isso não reproduz exatamente uma trajetória interrompida. Os mesmos critérios de compatibilidade da bancada são aplicados. As sessões do terminal aparecem na aba **Sessões** da UI quando estão em `runs/experiments/`.

Para mudar tarefa, recompensas ou hiperparâmetros de uma política existente, use transferência:

```bash
python train.py --preset solo_goal --init-from runs/experiments/solo-cli/latest.pt --learning-rate 0.0001 --reward approach=1.5 --steps 100000 --no-live
```

`--save CAMINHO.pt` escolhe outro destino para os pesos; `--log CAMINHO.jsonl` copia as métricas ao encerrar. Na pasta da sessão ficam configuração, métricas, episódios, status e controles. `output.log` é criado pelo launcher gráfico; no terminal, redirecione stdout/stderr se quiser guardar também a saída textual.

## Pausar e encerrar

**Ctrl+C** solicita parada cooperativa, atualiza o rollout parcial e salva os pesos antes de sair. Aguarde a confirmação com o caminho do checkpoint. Também é possível enviar comandos a partir de outro terminal:

Para execução longa, `--max-hours 8` ou `--until 2026-10-22T07:00:00-03:00` define um limite de tempo real com salvamento cooperativo. O horário absoluto precisa incluir fuso e estar no futuro. No Windows, `--keep-awake` evita suspensão automática enquanto o processo estiver treinando, sem manter a tela ligada nem alterar a configuração global de energia. Veja [1v1 e treino noturno](OVERNIGHT_1V1.md).

```bash
python train.py --status runs/experiments/solo-cli
python train.py --pause runs/experiments/solo-cli
python train.py --continue runs/experiments/solo-cli
python train.py --stop runs/experiments/solo-cli
```

Esses comandos enviam pedidos ao worker por `control.json`. A resposta confirma o pedido; `--status` permite verificar quando foi aplicado. **Continue** mantém o mesmo processo e estado pausado; **resume** inicia outra sessão a partir de um checkpoint. O estado em disco pode permanecer antigo se o processo for encerrado à força.

## Avaliar e integrar com scripts

```bash
python train.py --eval-only --resume runs/experiments/solo-cli/latest.pt --eval-episodes 20
python evaluate.py --policy runs/experiments/solo-cli/latest.pt --episodes 20
python train.py --preset solo_goal --steps 100000 --no-live --output json
python train.py --preset solo_goal --steps 100000 --run-dir runs/experiments/solo-quiet --no-live --quiet
```

`--eval-only` avalia a política congelada no exercício do checkpoint e imprime JSON. `evaluate.py` também compara com ações zero e bot tático. `--output json` gera eventos JSONL (`configuration`, `progress`, `metrics`, `saved` e `error` quando houver falha). `--quiet` silencia stdout durante o treino; erros continuam em stderr e retornam código de saída diferente de zero. `--print-config` imprime o JSON explicitamente solicitado mesmo com `--quiet`.

## Instalação em servidor sem interface gráfica

Use um ambiente virtual separado para instalar as dependências headless:

```bash
python -m venv .venv-headless
# Linux/macOS:
source .venv-headless/bin/activate
# Windows PowerShell: .\.venv-headless\Scripts\Activate.ps1
python -m pip install -r requirements-headless.txt
python train.py --preset solo_goal --steps 100000 --no-live
```

Esse conjunto usa `opencv-python-headless` e dispensa Pygame e Bluetooth. A instalação desktop existente também roda a CLI. As variantes `opencv-python` e `opencv-python-headless` compartilham o módulo `cv2`; mantenha as duas instalações em ambientes separados.
