# Validação

## Bancada de treino configurável — revisão atual

Executada em 21/10/2026. **74 testes aprovados**, Ruff, formatação, compilação dos módulos e diff sem erros. O [guia da bancada](TRAINING_WORKBENCH.md) descreve as operações e seus limites.

As verificações novas incluem presets, quantidade real de corpos e observações em solo e times assimétricos, chegada sem gol versus gol físico, pesos de recompensa, rejeição de gol contra como sucesso, persistência dos parâmetros, self-play sem fallback tático na coleta e transferência de pesos sem reutilizar o otimizador. Eventos sintéticos de mouse e teclado confirmam que o preset e a edição de learning rate chegam à configuração enviada ao worker.

Um teste com processo Python real inicia coleta solo, lê seus quadros, pausa sem avançar as transições, continua, encerra cooperativamente e recarrega o checkpoint com otimizador. Também confirma que o benchmark lê a tarefa e a quantidade de robôs do checkpoint. A interface completa foi executada com SDL dummy, incluindo início de treino, frames reais de coleta, gráficos, parâmetros e encerramento com salvamento.

Uma execução de integração da janela com **Solo · ir ao disco**, dois ambientes, episódios de dois segundos e ritmo 5× foi encerrada ao fechar a janela: salvou 1.214 transições e 38 atualizações. A retomada preservou os contadores, adicionou 32 transições e chegou à atualização 39. Essa execução curta existe para testar o fluxo da interface, não para produzir uma política pronta.

Avaliação posterior da tarefa de chegada, seeds 10000–10001 e limite de dois segundos por episódio:

| Controlador | Sucessos de chegada | Derrotas | Cortes por tempo |
|---|---:|---:|---:|
| Ações zero | 0 | 0 | 2 |
| Bot tático | 2 | 0 | 0 |
| PPO da integração curta | 0 | 0 | 2 |

Sucesso de chegada não equivale a gol. O teste curto não demonstrou que o PPO aprendeu a tarefa. O treino e a avaliação usam configurações persistidas; a visualização identifica prévia tática, coleta exploratória, avaliação congelada e replay separadamente. Não houve acionamento de hardware ou validação óptica real.

## Histórico: referência comercial HockeyBot

Em 21/10/2026, a geometria foi substituída pela referência comercial `robocore-hockeybot-stock-photo-v1`. Os dados publicados e as hipóteses estão em [HOCKEYBOT_STOCK.md](HOCKEYBOT_STOCK.md). O contorno foi reconstruído pela foto do fabricante; não houve medição do chassi físico.

- **57 testes aprovados.** A suíte foi ampliada para verificar chassis/CAD idênticos entre papéis, envelope incluindo pneus, abertura real do recorte, impactos nos pneus, colocação sem penetração profunda e contatos rasos na cavidade.
- Imagens sintéticas verificam que o disco continua detectável na abertura frontal, enquanto manchas sobre a tampa ou os pneus são rejeitadas. Os testes existentes de tracking sim/real continuam aplicáveis.
- Checkpoints de outro modelo físico são rejeitados; o transporte comercial original é distinguido do adaptador próprio. Nenhuma chamada de teste acionou Bluetooth real.
- Interface e vista ampliada renderizadas com SDL dummy. CAD OBJ/STL regenerado a partir do mesmo contorno, com integridade das malhas verificada nos testes.
- Ruff, compilação dos módulos e `git diff --check` aprovados. A CLI rejeitou o checkpoint da revisão anterior por incompatibilidade de modelo físico, como esperado.

Passagem de treino com o novo corpo:

```bash
python train.py --steps 1024 --envs 2 --rollout 64 --episode-steps 180 --eval-episodes 1
```

Completou 1.024 transições de ambiente, 3.072 amostras individuais e oito atualizações em aproximadamente 5,7 s neste executor. KL final 0,00214, clipping 0,0234, métricas finitas. A única avaliação reservada terminou por tempo, sem gol azul; isso verifica funcionamento e **não demonstra uma política competente**. Os resultados de 20.000 passos abaixo pertencem ao corpo anterior e não devem ser atribuídos ao HockeyBot desta revisão.

## Histórico: revisão v2 com o corpo anterior

Executada em 21/10/2026 sobre a revisão proposta. Plataforma Linux, Python 3.12.14, PyTorch 2.14.1 CPU, NumPy 2.3.5, OpenCV 5.0.0.93, pygame 2.6.1, Gymnasium 1.4.0 e Pymunk 7.x.

## Verificações funcionais

- **43 testes aprovados**, incluindo GAE por agente/ambiente, bootstrap de truncamento, razão de probabilidade unitária sem atualização, compatibilidade de checkpoint, sementes reproduzíveis, equivalência sim/real, convenção ArUco, velocidade/rotação estimadas, expiração de tracking, detector do disco e supervisor comum.
- Contatos verificados: inércia altera a resposta angular; disco rebate e transfere momento; penetração inicial é resolvida; robôs respeitam paredes com orientação; gol exige cruzamento completo; coeficiente de restituição configurado corresponde ao par de materiais.
- Ações pequenas usam o mesmo mapa PWM no simulador e no adaptador físico. Testes de rádio usam cliente falso, sem transmitir: limite de frequência, deadline, CRC, STOP e endereços únicos.
- O parser C++ do firmware foi compilado nativamente e decodificou 30 pacotes aleatórios produzidos pelo Python, incluindo sinais, limites e sequência; pacotes corrompidos foram rejeitados.
- Firmware ESP32 compilado com PlatformIO espressif32 6.10.0 e Arduino 2.0.17, usando **pinagem de teste**, sem gravação ou conexão com hardware.
- `ruff check .`, `ruff format --check .`, compilação dos módulos e `git diff --check` aprovados.
- Interface renderizada com SDL dummy; processo de treino separado e modo de gravação da visão sem BLE verificados. Não houve teste humano de jogabilidade nem uso de câmera física.

## Execução de treinamento

```bash
python train.py --steps 20000 --envs 4 --rollout 128 --episode-steps 450 --eval-episodes 5
```

Seed 42, currículo e adversários mistos. Concluiu **20.000 transições de ambiente, 60.000 amostras individuais e 40 atualizações**, em aproximadamente 77 segundos neste executor. Métricas finais finitas: KL aproximado 0,00481, fração de clipping 0,0590 e entropia estimada 0,791.

Uma execução curta separada com `--opponent selfplay` completou oito atualizações e exercitou troca entre o bot e snapshots congelados. Isso verifica a execução do caminho, não a qualidade de uma liga de self-play.

## Avaliação reservada

```bash
python evaluate.py --episodes 5 --max-steps 450 --policy CAMINHO_DO_CHECKPOINT --output runs/benchmark.json
```

Seeds 10000–10004, perfil Micro padrão, episódios de até 15 segundos, mesma inicialização por controlador:

| Azul | Vitórias | Derrotas | Limites de tempo |
|---|---:|---:|---:|
| Ações zero | 0 | 2 | 3 |
| Bot tático | 0 | 0 | 5 |
| PPO após o treino curto | 0 | 2 | 3 |

**O treino curto não produziu uma política competitiva.** Ele demonstra que o algoritmo executa atualizações coerentes sem os erros identificados na auditoria. Os pesos de validação não são distribuídos como uma IA pronta. A demonstração inicial usa bots táticos e identifica o controlador em uso.

Os resultados não são diretamente comparáveis aos do simulador antigo: arena, contatos, observações, frequência e recompensas foram corrigidos. Também não constituem estatística de vitória em partidas completas de cinco minutos. Use mais seeds, episódios completos e cenários de bancada para avaliar futuros checkpoints.

## Limites que continuam exigindo bancada

Puck, motores, atritos, inércia e latências ainda são parâmetros de projeto. Faltam medições do seu equipamento para declarar fidelidade sim-to-real. Não houve gravação de firmware, acionamento de motores, calibração óptica real nem execução dentro do Autodesk Fusion. A aceitação de autonomia e câmera externa deve ser conferida no evento específico.
