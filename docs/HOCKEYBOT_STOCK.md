# Referência: HockeyBot comercial RoboCore

O hardware alvo é o HockeyBot pronto, comandado pelo aplicativo RoboCore Joystick via Bluetooth. Atacante, ala e goleiro usam **o mesmo robô**; os papéis alteram o controlador. O modelo `robocore-hockeybot-stock-photo-v1` substitui os três chassis inventados anteriormente. Consulta das referências: 21/10/2026.

## Dados publicados e hipóteses

| Grandeza | Valor usado | Procedência |
|---|---:|---|
| Massa montada | aproximadamente 360 g | Resposta do fabricante de 17/03/2026 na [página do HockeyBot](https://www.robocore.net/kits-didaticos/hockeybot/preto) |
| Motorização de reposição | JGA16-050, 6 V, redução 45:1 | Fabricante indica o [motor 6 V / 266 rpm](https://www.robocore.net/motor-caixas-de-reducao/motor-com-caixa-de-reducao-6v-266rpm) na página do robô |
| Rotação sem carga / nominal | 266 / 200 rpm a 6 V | Especificação do motor; tensão nominal não comprova a tensão real da bateria/ponte H |
| Torque de bloqueio / nominal | 1 / 0,43 kgf·cm | Especificação do motor; não é torque contínuo disponível na roda |
| Pneu | diâmetro externo 60 mm, largura 20 mm | [Pneu NEO 60 mm](https://www.robocore.net/roda/pneu-neo-60mm), indicado como reposição pelo fabricante |
| Material / massa de cada pneu | neoprene expandido, Shore A 20 / 13 g | Página do pneu; massa não inclui cubo/roda |
| Envelope montado | 164 × 144 mm | **Estimativa visual**, inclui pneus; não é medida oficial |
| Bitola / posição do eixo | 118 mm / x = −52 mm | **Estimativa visual**, entre centros dos pneus; origem descrita abaixo |
| Apoios frontais | x = +68 mm, y = ±57 mm | Dois contatos equivalentes nas pontas da base; hipótese, sem apoio medido |
| Resposta de motor / deadband | 0,12 s / 12% | Hipóteses; precisam de ensaios de degrau com o firmware original |
| Atritos pneu / apoio | 0,85 / 0,08 | Hipóteses; Shore A não determina coeficiente de atrito |

Velocidade linear sem carga derivada: π × 6 × 266 / 60 = **83,57 cm/s**. A rotação nominal daria 62,83 cm/s; nenhuma delas é uma medição no piso. O simulador usa uma curva DC linear de torque versus velocidade, limitada pelo torque de bloqueio e pela aderência. A resposta filtrada, a frenagem ao comando zero e as assimetrias são aproximações, não o comportamento identificado do controlador original. Não há modelo de bateria, corrente, aquecimento ou folga de engrenagens.

## Contorno e coordenadas

A referência visual é a [foto superior do fabricante, 3388_4_H, 600 × 600 px](https://d229kd5ey79jzj.cloudfront.net/3388/images/3388_4_H.png), ligada ao produto. Ela mostra rodas expostas na traseira, tampa elevada e uma base frontal com dois lobos curvos no recorte para o disco.

`core/geometry.py` guarda os pontos da metade direita da foto e os espelha em x = 299,5 px. Isso evita introduzir assimetrias de perspectiva no robô. A transformação é `x_local = (y_pixel − 295) × comprimento / 486`, `y_local = (299,5 − x_pixel) × largura / 474`. No robô, +x aponta para a frente e +y para a esquerda. A origem fica no centro do envelope montado, incluindo pneus, e também é a hipótese de centro de massa; não é um centro de massa medido.

As escalas horizontal e vertical são independentes e aproximadas: a foto tem perspectiva e peças em alturas diferentes. As dimensões conhecidas dos pneus ajudam a estimar a proporção, mas **não tornam essa imagem uma vista ortográfica calibrada**. O envelope e a bitola são valores iniciais de modelagem. Trate-os como sujeitos a erro da ordem de 10% ou maior; esse número é uma cautela de modelagem, não uma tolerância experimental nem intervalo estatístico. Para fidelidade dimensional, substitua por paquímetro, foto ortogonal calibrada ou CAD oficial.

O recorte permanece aberto na decomposição convexa usada pelo Pymunk. Os dois pneus de 60 × 20 mm têm seus próprios colisores. A renderização e o CAD leem os mesmos contornos; as rodas não são círculos decorativos. A placa CAD é plana, com espessura hipotética de 3 mm, sem reproduzir a rampa, furos, suportes ou montagem. O modelo 2D não reproduz o disco subindo na rampa, inclinação, tombamento ou flexão do pneu.

A inércia distribui a massa residual sobre a placa e adiciona 13 g em cada posição de pneu. Cubos, motores, eletrônica e bateria não foram pesados separadamente; essa distribuição não é uma montagem física identificada. `RobotConfig.inertia` permite substituir a aproximação por valor medido. As cargas nos apoios pressupõem centro de massa na origem.

## Câmera e Bluetooth originais

O [tutorial oficial](https://www.robocore.net/tutoriais/hockeybot) descreve o RoboCore Joystick e o comando de avanço/ré e direção. O protocolo GATT, os pacotes e o modo de frenagem não foram identificados nas fontes consultadas. O firmware `hockey_v2` do repositório é um adaptador próprio para eletrônica compatível e **não implementa o Bluetooth original**.

`configs/robot.example.json` identifica `transport: robocore-stock`. A câmera, gravação e inferência dry-run funcionam nesse modo; `--armed` é rejeitado antes de abrir câmera ou rádio. O transporte `hockey-v2-custom` exige escolha explícita e verificação do adaptador. A ponte própria não deve ser apresentada como integração pronta com os robôs comerciais ou motivo para substituir seu firmware.

Para o hardware original, a próxima integração depende de documentação/captura autorizada do protocolo e ensaios físicos de sentido, PWM, latência e parada. As tags ArUco são uma adição proposta para a câmera zenital; não são sensores ou marcações de fábrica. O PPO recebe poses, velocidades estimadas, distâncias relativas e validade do tracking, sem acesso privilegiado aos motores. A geometria física consta nos checkpoints; pesos treinados com o antigo chassi são rejeitados na interface, treino, avaliação e runner.

O detector exclui o contorno do robô e dos pneus, preservando a abertura frontal; não usa mais um círculo de exclusão em torno do centro. `puck_hsv_lower` e `puck_hsv_upper` permitem ajustar a cor do disco no JSON. Os padrões detectam um disco escuro. O mini puck colorido fotografado com o produto exige configuração e ensaios próprios; não foi presumido idêntico ao disco Micro de 3,17 cm usado no perfil padrão. Cor semelhante à do chassi, oclusões, reflexos e perda de separação durante contato ainda podem exigir outro detector/marcação. A simulação de detecções com ruído não comprova desempenho em pixels reais.

## Arena e categoria

O [kit comercial](https://www.robocore.net/kits-didaticos/kit-hockeybot/azul) é divulgado para a categoria HockeyBot. Isso não demonstra que a arena dessa categoria tenha as medidas da categoria Micro Hockey. O perfil de arena continua separado do hardware; `micro-2024` é um perfil regulamentar Micro, não uma alegação sobre o campo oficial de HockeyBot. Confira a categoria e a edição do evento antes de substituir qualquer medida de arena.

Para gerar a vista de referência e os modelos:

```bash
python -m ui.robot_renderer --output runs/hockeybot-reference.png
python cad/fusion_chassis_generator.py
```

Veja [o protocolo de medição](COMPETITION_AND_CALIBRATION.md) para transformar essa referência em um modelo validado do seu equipamento.
