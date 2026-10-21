# Pista com gols recuados e treino solo progressivo

## Geometria da pista de referência

O perfil `micro-practice` reproduz a estrutura da referência fornecida: cantos arredondados e um gol retangular recuado em cada extremidade curta. Cada gol tem duas paredes laterais e uma parede de fundo. A boca permanece aberta para a pista. A linha tracejada é somente a referência de pontuação, sem colisão.

| Medida | Valor usado | Origem |
|---|---:|---|
| Área retangular de jogo | 300 × 150 cm | Escolha dentro da faixa Micro |
| Abertura do gol | 52 cm | Regulamento Micro |
| Altura do gol | 18 cm | Regulamento Micro; a física é 2D |
| Raio dos cantos | 12,5 cm | Diâmetro de 25 cm, dentro da faixa Micro |
| Profundidade de cada recuo | 26 cm | Estimativa de bancada para a referência; não é uma medida extraída do regulamento nem da foto |
| Comprimento externo incluindo os dois recuos | 352 cm | 300 + 2 × 26 cm |

A referência regulamentar é [RoboCore Hockey Rules, revisão de 15/01/2024, seção 3.2](https://robocore-eventos.s3.sa-east-1.amazonaws.com/public/RoboCore_Hockey_Rules.pdf). Ela informa a faixa de 293–366 cm de comprimento e 146–183 cm de largura, mas não fixa a profundidade do gol. O perfil Micro não identifica automaticamente a categoria e as medidas de um evento HockeyBot. Confirme as medidas da pista construída para ajustar o modelo.

`micro-2024` mantém a profundidade anterior de 12 cm. `micro-practice` é uma alternativa explícita, com 26 cm, e pode ser escolhida na UI ou por `--profile micro-practice`. O desenho e os segmentos de colisão são gerados pelo mesmo contorno em `core/arena_geometry.py`. O disco só marca depois de cruzar inteiro a linha, com todo seu diâmetro entre as traves. As paredes contêm disco e robôs por contato físico; o desenho não cria uma barreira invisível na boca.

## A recompensa é variável?

Os **pesos** são parâmetros editáveis e ficam fixos ao longo de uma sessão. A **recompensa calculada** varia em cada passo. Não existe uma tabela que aumenta os bônus automaticamente conforme o robô aprende. O currículo altera as posições iniciais conforme o sucesso, mantendo o objetivo e os mesmos pesos.

Os presets `solo_goal` e `solo_curriculum` usam:

| Componente | Peso padrão | Como participa |
|---|---:|---|
| Gol | 20 | +20 a favor; −20 por gol contra |
| Progresso do disco | 3 | Diferença de potencial até o gol adversário |
| Aproximação | 0,4 | Diferença de potencial da distância robô–disco |
| Tempo | 0,006 | Subtraído a cada passo de controle, a 30 Hz |
| Esforço | 0,001 | Subtraído proporcionalmente à média das ações ao quadrado |
| Bônus de chegada | 10 | Aplicado somente ao objetivo `reach` |
| Distância de chegada | 10 cm | Limite de sucesso somente no objetivo `reach`; não é um peso de recompensa |

Progresso e aproximação têm a forma `peso × (gamma × potencial seguinte − potencial anterior)`. O potencial seguinte é zero em um término real; no limite de tempo, o valor futuro continua sendo estimado pelo PPO. Por causa do desconto, um disco parado pode gerar um termo negativo. Esses componentes não são bônus fixos por centímetro ou por contato. Chegar perto não conclui o objetivo de marcar gol, e permanecer encostado no disco não dá pontos por toque.

Cada `step` devolve os termos ponderados e com sinal em `info.reward_components`, e o acumulado do episódio em `info.episode_reward_components`. A soma corresponde ao retorno médio dos robôs azuis. No solo, é o retorno do único robô. A aba **Recompensas** mostra o acumulado da coleta ao vivo sob cada componente. `episodes.jsonl` guarda os acumulados por episódio; `metrics.jsonl` guarda `mean_reward_components`, a média nos últimos vinte episódios. A CLI imprime esses componentes nas atualizações ou os inclui nos eventos JSONL.

Para mudar pesos mantendo uma rede aprendida, encerre a sessão e use transferência, com novo otimizador. Retomada conserva a identidade da configuração e rejeita recompensas incompatíveis.

## Currículo solo por desempenho

Escolha **Solo · progressivo até o gol** na UI ou use `--preset solo_curriculum`. O preset seleciona `micro-practice`, um robô azul, nenhum vermelho, objetivo de marcar gol, oito ambientes e orçamento inicial de um milhão de transições. Esses valores são pontos de partida editáveis, não uma garantia de competência.

| Etapa | Inicialização | Sucesso |
|---|---|---|
| 1 · Finalizar | Disco a 8–30 cm da linha; robô atrás e aproximadamente alinhado | Gol real |
| 2 · Conduzir | Disco mais longe e deslocado lateralmente; robô começa atrás | Gol real |
| 3 · Localizar e marcar | Posições variadas, aproximação de qualquer direção e orientação aleatória | Gol real |

O robô não avança apenas porque gastou passos. A promoção exige pelo menos **30 episódios concluídos na etapa atual** e taxa de sucesso de pelo menos **70%**, considerando os últimos **50 episódios dessa etapa**. Os três parâmetros podem ser editados na UI e pela CLI. A mudança é aplicada no reset de cada ambiente, conservando a trajetória em curso. Não há adversários escondidos e o disco continua livre na física.

A avaliação reservada usa sempre a **etapa 3**, com ações determinísticas e seeds reservadas, mesmo quando a coleta está na etapa 1. Assim, os gols de finalização não são apresentados como evidência de que a política já domina a busca completa. Status, métricas e checkpoints registram a etapa do currículo. A retomada conserva essa etapa e o otimizador; reinicia episódios e a janela de evidências de promoção.

```powershell
# UI
.\.venv\Scripts\python.exe main.py --preset solo_curriculum

# Headless
.\.venv\Scripts\python.exe train.py --preset solo_curriculum --steps 1000000 --no-live

# Personalizar critérios de promoção
.\.venv\Scripts\python.exe train.py --preset solo_curriculum --curriculum-success-threshold 0.8 --curriculum-min-episodes 40 --curriculum-window 60 --no-live

# Usar o JSON equivalente
.\.venv\Scripts\python.exe train.py --config configs/training/solo_curriculum.json --no-live

# Avaliação comparativa após o treino
.\.venv\Scripts\python.exe evaluate.py --policy runs/experiments/SESSAO/latest.pt --episodes 20
```

## O que falta para competência no robô real

A localização usa o contrato compartilhado de câmera zenital, rastreamento do disco e poses dos robôs. A política recebe as observações estimadas, com atraso, ruído e perdas simuladas. As recompensas podem usar o estado físico durante o treino; o ator continua sem receber posse verdadeira ou velocidade real das rodas.

Para avaliar aprendizado, acompanhe os gols reais, a taxa de sucesso da **avaliação completa**, os componentes do retorno e o tempo parado. Compare com ações zero e bot tático, em várias seeds. Concluir o orçamento ou aumentar a recompensa de coleta não comprova uma política competente.

Massa, atrito, resposta dos motores, dimensões da pista e latência da câmera precisam ser calibrados no equipamento. A integração Bluetooth original do HockeyBot continua pendente conforme [HOCKEYBOT_STOCK.md](HOCKEYBOT_STOCK.md); as alterações de arena e currículo não resolvem esse transporte. Um teste curto de execução valida o software, não o desempenho autônomo na pista física.

## Ensaio local de desenvolvimento — 21/10/2026

A suíte passou com **112 testes e 2 ignorados**, incluindo gol com travessia real do disco, contenção pelo fundo do recuo, spawns sem sobreposição, soma dos componentes de recompensa, promoção por desempenho e retomada da etapa do currículo.

Um ensaio com seed 42 acumulou **65.536 transições e 32 atualizações PPO**, em duas execuções de 8.192 e 57.344 passos. Foram usados oito ambientes, episódios curtos de cinco segundos, randomização habilitada e os pesos padrão. O worker avançou de finalização para condução após atingir o critério de promoção.

| Coleta exploratória | Episódios concluídos | Gols |
|---|---:|---:|
| Finalização | 447 | 99 |
| Condução | 43 | 1 |

Esses totais incluem o período inicial sem aprendizagem e não equivalem à taxa recente usada para promoção. Na avaliação determinística da busca completa, houve **0 gols em 2 episódios de cinco segundos**. O ensaio mostra aprendizagem inicial de finalização e persistência do currículo, mas não comprova domínio da condução, busca completa ou transferência para o robô físico.

Os pesos locais estão em `runs/experiments/solo-curriculum-learning-check/latest.pt`; o resumo fica em `runs/solo-curriculum-validation-summary.json`. A pasta `runs/` é ignorada pelo Git. Para continuar esse ensaio com orçamento maior e episódios de trinta segundos:

```powershell
.\.venv\Scripts\python.exe train.py --resume runs/experiments/solo-curriculum-learning-check/latest.pt --steps 1000000 --episode-seconds 30 --no-live
```
