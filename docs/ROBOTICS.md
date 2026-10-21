# Operação com visão de cima

Monte uma câmera fixa que veja todo o campo, seis IDs e o disco. Evite exposição automática que borre movimento. Fixe ArUco DICT_4X4_50 com a aresta superior apontando para a frente: azul 0,1,2; vermelho 3,4,5. Meça offset da tag em relação ao centro do chassi; esta versão pressupõe tag centralizada. Ajuste a montagem ou transforme a pose antes do construtor de observação se ela estiver deslocada.

O JSON de câmera informa o perfil, pontos de calibração TL/TR/BR/BL e atraso de captura medido. Opcionalmente informe `camera_matrix`, `dist_coeffs` e `robot_corners_px` para calibrar o plano elevado das tags. Os pontos de homografia devem vir da imagem sem distorção. Não use os cantos arredondados como se fossem os vértices retangulares: use pontos de referência mensuráveis que representem o retângulo métrico.

O estimador é comum à câmera e às detecções simuladas. Estados têm último timestamp observado, velocidade e validade; após 250 ms não alimentam o ator como posição válida. O supervisor é mais conservador: os três robôs próprios e o disco devem estar visíveis/recentes em até 150 ms para permitir movimento. Os adversários podem ser temporariamente desconhecidos, com confiança zero na observação.

`run_robot.py` usa um thread de captura com slot do frame mais recente, percepção/inferência e controle a 30 Hz. O log registra validade e comandos. `cap.read()` oferece timestamp no host; subtraia `capture_delay_seconds` medido. Dry-run não conecta ao rádio. O runner rejeita configuração física diferente da usada no treinamento.

## Firmware e protocolo

Para coletar dados antes de ter pesos treinados, execute `run_robot.py --config ... --track-only --preview`. O JSONL registra poses e velocidades estimadas; os dados brutos da calibração continuam necessários. Esse modo não aceita `--armed`.

O firmware em `firmware/hockey_v2` é um adaptador próprio, não o firmware de fábrica da RoboCore. Verifique a placa exata e o circuito de acionamento. O exemplo pressupõe ponte H controlável por dois PWM por motor; outros drivers precisam de adaptação. A pinagem é intencionalmente obrigatória em `include/pins.h`; o exemplo não informa pinos reais. A compilação de CI usa apenas uma pinagem de teste e não produz configuração pronta para gravação.

BLE: serviço `6e400001-b5a3-f393-e0a9-e50e24dcca9e`, RX com final `0002`, TX de leitura com final `0003`. Antes de enviar, o host exige que TX retorne `HOCKEY-V2`.

Pacote de 13 bytes, little-endian:

Para compilar, instale PlatformIO (`pip install platformio`), crie `firmware/hockey_v2/include/pins.h` com os pinos verificados e execute `pio run --project-dir firmware/hockey_v2`. O arquivo `tests/fixtures/compile_pins.h` serve exclusivamente à compilação de CI; não é uma pinagem para gravação no robô.

| Offset | Campo |
|---|---|
| 0–1 | AA 55 |
| 2 | Versão 2 |
| 3–4 | Sequência uint16, com wraparound |
| 5–6 | PWM esquerdo int16, −255…255 |
| 7–8 | PWM direito int16, −255…255 |
| 9–10 | Validade uint16, 20…200 ms |
| 11–12 | CRC16-CCITT sobre bytes 0–10, inicial FFFF |

Comandos antigos e CRC inválido não renovam a validade. O microcontrolador zera as saídas ao vencer o TTL e ao desconectar, independentemente do programa Python. STOP é aceito mesmo com sequência repetida. A implementação Python limita envios regulares a 30 Hz; emergência ignora a espera. A conversão ação→PWM é compartilhada com a simulação.

## Sequência de validação

Primeiro teste firmware com rodas suspensas, comandos conhecidos e parada por desconexão/TTL. Depois valide IDs, endereços e sentido das rodas, com velocidade limitada e espaço livre. Compare o tracking com posições medidas. Use dry-run com uma política congelada e inspecione logs. Somente depois habilite `protocol_verified` e `--armed`.

Não houve validação deste firmware no seu robô, do rádio ou da óptica nesta alteração. A identificação mecânica e o ensaio da parada física continuam necessários.
