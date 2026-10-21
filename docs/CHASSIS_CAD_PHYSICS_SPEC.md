# Chassi, CAD e modelo físico

O modelo agora toma como referência o **HockeyBot comercial controlado pelo RoboCore Joystick**. Massa, pneus e motorização têm referências do fabricante; contorno e bitola são estimativas por foto, não desenhos cotados oficiais. Veja a [procedência de cada parâmetro](HOCKEYBOT_STOCK.md). O limite Micro de 20 × 20 cm e 454 g não descreve a massa ou o formato do produto.

`core/geometry.py` define o contorno usado pelo motor físico, interface e gerador CAD. O recorte frontal curvo é decomposto em polígonos convexos sem preencher sua cavidade. Os dois pneus expostos também têm colisores. Os três papéis usam contornos idênticos. Não há retenção por código nem gatilho de chute; puck e robôs trocam impulsos por contatos.

A origem fica no centro do envelope montado. As rodas aplicam forças em x = −5,2 cm, y = ±5,9 cm; os dois contatos frontais equivalentes ficam em x = +6,8 cm, y = ±5,7 cm. Essas posições são estimadas. O modelo de inércia distribui massa residual no contorno e adiciona a massa dos pneus; o centro de massa é assumido na origem. Substitua por inércia medida em `RobotConfig.inertia` e calibre a distribuição real. A curva de motor usa dados nominais de 6 V, sem representar bateria ou eletrônica originais.

Execute `python cad/fusion_chassis_generator.py` para regenerar as bases de 3 mm nos formatos OBJ e STL e seus metadados. O STL não guarda unidades: importe em **centímetros**. A triangulação trata o contorno côncavo e produz superfícies fechadas verificadas por teste.

O script Fusion lê o mesmo JSON e cria três bases idênticas extrudadas. Ele não cria montagem completa, rampa, furos, alojamentos de motor, tampa ou bateria. A espessura de 3 mm é hipótese de exportação. Os metadados incluem a posição dos pneus, mas os STL/OBJ representam apenas a placa plana. A massa aproximada de 360 g é a referência do robô montado, não a massa calculada da base. Estes arquivos não são CAD para fabricação de uma réplica.

Veja [competição e calibração](COMPETITION_AND_CALIBRATION.md) para procedência, medições e limitações.
