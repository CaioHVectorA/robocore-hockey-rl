# Recuperar fundamentos antes do self-play

A análise do 1v1 de 21–22/10 encontrou estagnação, adversários frágeis e vitórias do PPO sem tocar no disco. O preset `solo_foundation` é um experimento de recuperação; não autoriza concluir que o problema está resolvido apenas porque o código roda.

## Recompensas candidatas

| Componente | Peso | Significado |
|---|---:|---|
| Gol marcado / sofrido | +20 / −20 | Placar real, inclusive gols contra do adversário |
| Progresso | 3 | Potencial do disco até o gol |
| Aproximação | 2 | Potencial de proximidade robô–disco |
| Alinhamento | 1 | Potencial limitado: perto, atrás do disco e orientado para o gol |
| Primeiro contato | 1 | Pago uma única vez por episódio |
| Condução | 3 | Novo máximo de progresso do disco após toque azul recente, com último toque azul |
| Tempo | 0,006 | Custo por passo a 30 Hz |
| Esforço | 0,0002 | Custo proporcional à média das ações ao quadrado |
| Inatividade | 0,003 | Custo por passo após três segundos abaixo de 2 cm/s, a mais de 20 cm do disco |

O bônus de chegada permanece exclusivo de `objective=reach`; não é aplicado ao objetivo de marcar gol. Progresso, aproximação e alinhamento usam diferenças de potencial com desconto. Alinhamento não é bônus repetido por manter uma pose. Primeiro contato é limitado a uma ocorrência; condução paga apenas avanço acima do máximo anterior do episódio, até 0,5 s após toque azul e com último toque azul, sem permitir recuperar o mesmo bônus indo e voltando. Esse máximo é atualizado também quando o adversário move o disco.

Os campos novos têm peso zero nas configurações antigas. Os pesos são fixos na sessão e editáveis na UI e por `--reward NOME=VALOR`. Não há posse artificial, teleporte de disco ou controlador tático oculto no ator.

## Currículo e proteção contra estagnação

O currículo mantém finalização perto do gol, condução e busca de qualquer direção. No preset novo, promoção exige tanto 70% de sucesso ativo em pelo menos trinta episódios da etapa (janela de cinquenta) quanto 70% em dez episódios determinísticos de validação da etapa, seeds 30000–30009. A primeira prova ocorre na próxima atualização após atingir o critério de coleta, sem esperar o intervalo da avaliação completa e treinar demais a etapa fácil. Se falhar, aguarda o intervalo de avaliação antes de repetir; uma prova incompleta não promove. A avaliação regular usa busca completa, seeds 10000–10009, e continua separada da prova da etapa.

Essas seeds fixas são validação para desenvolvimento, não um teste final intocado. Use outra faixa de seeds para decidir sobre competência e transferência ao físico. O preset continua sem adversário; não avança automaticamente para 1v1.

`stagnation_patience=200` salva e encerra após duzentos episódios consecutivos sem sucesso ativo na mesma etapa. O status informa `stop_reason=stagnation`. O contador é preservado ao retomar. Isso detecta ausência de sucesso; não detecta todo tipo de platô com vitórias ocasionais. `0` desativa.

O preset usa episódios de quinze segundos, oito ambientes, `gamma=0,999` e entropia `0,01`. Os valores são candidatos explícitos para testes curtos. Retomada conserva a tarefa e os parâmetros; mudar recompensas exige transferência com novo otimizador.

## Placar e habilidade são medidas diferentes

Logs de episódios, avaliações e métricas agora incluem participação azul, toques, distância média ao disco, fração de inatividade, perdas de tracking e último time a tocar antes do gol. `active_success` exige chegada no exercício `reach`, ou gol a favor com pelo menos um toque azul no episódio. Isso elimina vitórias sem participação; não é prova de autoria causal do gol nem de boa tática. O campo de último toque permite auditar gols contra/deflexões.

Novas sessões selecionam o melhor modelo pela taxa de sucesso ativo em avaliações completas. `point_score` continua informativo, sem promover empates por tempo. Ao retomar um checkpoint antigo, o limiar do seletor anterior é reiniciado porque a métrica mudou. Os arquivos antigos não são substituídos.

## Executar um experimento curto

```powershell
# Transferir da POC preservada, sem reutilizar o otimizador do 1v1 estagnado
.\.venv\Scripts\python.exe train.py --preset solo_foundation --init-from checkpoints/poc-solo-20261021.pt --steps 65536 --max-hours 0.25 --no-live

# Ou usar o JSON equivalente; a CLI cria uma nova pasta de sessão
.\.venv\Scripts\python.exe train.py --config configs/training/solo_foundation.json --init-from checkpoints/poc-solo-20261021.pt --steps 65536 --no-live

# Parâmetros opcionais
.\.venv\Scripts\python.exe train.py --preset solo_foundation --stagnation-patience 200 --curriculum-eval-gate --reward first_touch=1 --print-config
```

Verifique contatos e gols reais em seeds novas, em cada etapa. Para entrar em disputa, faça primeiro testes com adversário parado e só depois adversários ativos. Não compense falta de domínio rodando o mesmo experimento durante outra noite.

## Ensaio de desenvolvimento de 22/10/2026

Foram comparadas a tabela original e a candidata, ambas transferidas da mesma POC, com seed de treino 42 e os mesmos hiperparâmetros/currículo da recuperação. Não foi reutilizado o otimizador do self-play. A primeira rodada, 32.768 passos por braço, expôs excesso de treino da finalização enquanto aguardava a avaliação na décima atualização: ambos os modelos regrediram em condução. A prova de etapa foi antecipada para a primeira atualização após atingir o critério; se reprovar, mantém o intervalo antes de nova prova.

A segunda rodada executou 65.536 passos por braço. Ambos passaram finalização com 10/10 na validação da etapa, já na primeira atualização. Ambos permaneceram em condução, sem atingir o critério de promoção. Avaliação final determinística, doze seeds novas 50000–50011 por etapa, episódios de até quinze segundos:

| Modelo | Finalização | Condução | Busca completa |
|---|---:|---:|---:|
| POC de partida | 12/12 | 5/12 | 0/12 |
| Recompensas originais + currículo corrigido | 12/12 | 4/12 | 0/12 |
| Recompensas candidatas + currículo corrigido | 12/12 | 6/12 | 1/12 |

São gols reais com participação azul, sem adversário. Não compare os valores brutos de retorno entre tabelas diferentes. Doze episódios e uma única seed de treino não estabelecem superioridade estatística nem competência em 1v1. O modelo candidato continua experimental; seus pesos foram preservados em `checkpoints/foundation-candidate-20261022.pt`, com configuração, hashes e resumo no manifesto. A busca completa ainda não foi dominada, e não foi iniciado outro treino longo.

As pastas locais `runs/experiments/foundation-{baseline,candidate}[-v2]-20261022` preservam as quatro sessões. `runs/foundation-validation-20261022/holdout.json` registra a primeira rodada (seeds 40000–40011); `holdout-v2.json`, a segunda. A primeira faixa foi usada para desenvolvimento e não é um teste intocado após esse diagnóstico.
