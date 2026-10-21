# Chassi, CAD e modelo físico

Os modelos são propostas de projeto. Não são desenhos oficiais nem medições de um HockeyBot comercial. O limite Micro de 20 × 20 cm e 454 g não define modelo de motor, eletrônica ou geometria obrigatória.

`core/geometry.py` define o contorno usado pelo motor físico, interface e gerador CAD. O motor usa três polígonos convexos por chassi; o rebaixo frontal permanece aberto. Não há retenção por código nem gatilho de chute. Puck e robôs trocam impulsos por contatos, usando massa e inércia dos corpos.

O centro do chassi é a origem. As rodas aplicam forças em x = −2,5 cm, y = ±7,25 cm; o patim aplica arrasto em x = +6,8 cm. O modelo de inércia é uma aproximação de massa distribuída no contorno. Pode ser substituído por uma inércia medida em `RobotConfig.inertia`. Atrito e motores são estimativas pendentes de calibração.

Execute `python cad/fusion_chassis_generator.py` para regenerar as bases de 3 mm nos formatos OBJ e STL e seus metadados. O STL não guarda unidades: importe em **centímetros**. A triangulação trata o contorno côncavo e produz superfícies fechadas verificadas por teste.

O script Fusion lê o mesmo JSON e cria três bases extrudadas. Ele não cria montagem completa, furos de placa, alojamentos de motor, tampa ou bateria. Verifique a integração mecânica antes de fabricar. A massa de 454 g nos metadados é a hipótese do robô montado usada na simulação, não a massa calculada da base.

Veja [competição e calibração](COMPETITION_AND_CALIBRATION.md) para procedência, medições e limitações.
