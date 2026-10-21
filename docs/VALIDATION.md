# Validação da revisão v2

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
