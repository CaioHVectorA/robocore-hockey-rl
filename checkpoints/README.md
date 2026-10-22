# Checkpoints preservados

Os arquivos abaixo são cópias selecionadas para demonstração e retomada. Os checkpoints de novas sessões continuam em `runs/`, fora do Git.

| Arquivo | Conteúdo |
|---|---|
| `poc-solo-20261021.pt` | POC solo, 65.536 transições e 32 updates; finalização e condução iniciais |
| `duel-selfplay-20261021.pt` | Estado final do 1v1, 1.221.152 transições e 597 updates; rede, otimizador, âncora e pool de 12 adversários |
| `duel-selfplay-20261021.best.pt` | Melhor checkpoint da mesma sessão segundo `point_score` em avaliações completas |
| `foundation-candidate-20261022.pt` | Experimento de recuperação solo, 65.536 transições novas e 32 updates; está na condução, ainda sem domínio da busca |

Consulte os manifestos JSON para hashes, configuração e avaliações. A última avaliação completa do 1v1 foi de 12 episódios: 0 vitórias, 2 derrotas e 10 limites de tempo. `point_score` conta limite de tempo como meio ponto; não é taxa de vitória. Estes pesos não comprovam competência contra pessoas nem no robô físico.

## Retomar o 1v1

Na raiz do projeto, com o ambiente Python instalado:

```powershell
.\.venv\Scripts\python.exe train.py --resume checkpoints/duel-selfplay-20261021.pt --steps 100000000 --max-hours 8 --keep-awake --no-live --output json
```

A CLI cria uma nova pasta automaticamente. A configuração do checkpoint e o otimizador são recuperados; episódios e geradores de ambiente reiniciam. Não use o preset de transferência da POC para continuar o estado do 1v1.

## Jogar contra a política

```powershell
.\.venv\Scripts\python.exe main.py --training-config configs/training/duel_selfplay.json --policy checkpoints/duel-selfplay-20261021.pt --human
```

Você controla o vermelho com WASD/setas; espaço gira. A política controla o azul. Veja [o guia 1v1](../docs/OVERNIGHT_1V1.md) e [as instruções de acompanhamento](../docs/ANTIGRAVITY_TRAINING_HANDOFF.md).

O GIF da POC está em [`docs/assets/ppo-conducao-gol.gif`](../docs/assets/ppo-conducao-gol.gif). É um episódio selecionado de condução com gol, não uma avaliação estatística.

O antigo `hockey_ppo.pt`, já existente no repositório, é incompatível com a observação atual e não deve ser usado nesses comandos.

O modelo de recuperação de 22/10 usa recompensas novas, validação de etapa e parada por estagnação. Na bateria final de doze episódios por etapa, marcou 12 gols de finalização, 6 de condução e 1 de busca completa. É evidência inicial, com uma única seed de treino; não é um modelo pronto para competir. Veja [o protocolo e a comparação](../docs/FOUNDATION_TRAINING.md). Para continuar esse experimento em uma nova pasta:

```powershell
.\.venv\Scripts\python.exe train.py --resume checkpoints/foundation-candidate-20261022.pt --steps 65536 --max-hours 0.25 --no-live
```
