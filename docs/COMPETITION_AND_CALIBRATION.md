# Competição e identificação física

Referência dos perfis Micro: [RoboCore Hockey Rules, revisão 15/01/2024](https://robocore-eventos.s3.sa-east-1.amazonaws.com/public/RoboCore_Hockey_Rules.pdf), seções 2.2, 2.8, 3.2 e 4. Confira o regulamento do evento pretendido; este projeto não certifica elegibilidade.

| Parâmetro | Origem / tratamento |
|---|---|
| Arena Micro | Faixa regulamentar; o perfil padrão escolhe 300 × 150 cm dentro dela |
| Cantos | Perfil segue a redação de diâmetro 24–30 cm; confirme o desenho do evento, pois a relação de escala no texto merece esclarecimento |
| Gol | 52 cm de abertura e 18 cm de altura; profundidade de 12 cm é hipótese própria |
| Disco | Diâmetro 3,17 cm e espessura 1,27 cm; massa de 15 g é estimativa a substituir por pesagem |
| Chassi | Referência comercial HockeyBot: aproximadamente 360 g publicados; envelope de 16,4 × 14,4 cm e bitola estimados pela foto |
| Motores e pneus | Reposição indicada pelo fabricante: motor 6 V / 266 rpm e pneus NEO 60 × 20 mm; resposta real, atritos e assimetria ainda exigem medição |
| Rádio e câmera | Frequências, atrasos e perdas são hipóteses configuráveis; exigem medição no equipamento |

Uma regra define uma restrição; um parâmetro medido descreve um equipamento. Não use o limite de peso como comprovação de que o robô pesa exatamente esse valor. Confira ainda envolvimento e abertura da pá e se o formato de controle proposto é aceito pela organização.

O hardware alvo é o [HockeyBot comercial](HOCKEYBOT_STOCK.md). A categoria comercial HockeyBot e a categoria Micro não devem ser presumidas equivalentes. O perfil Micro descreve uma arena escolhida para simulação; confirme o campo e as regras específicos da edição em que os robôs prontos serão usados.

## Protocolo de bancada

1. Pese robô montado e disco. Meça contorno, bitola, posição do eixo em relação à marca visual, raio de roda e centro de massa. Registre instrumento, unidade, data e tolerância.
2. Calibre lente e homografia. Filme pontos de coordenadas conhecidas no piso e no plano das tags. Meça erro de posição/orientação em toda a arena, não só no centro.
3. Com rodas suspensas primeiro e depois no piso, aplique degraus de PWM em ambas as direções. Registre x, y, theta e timestamps. Estime deadband, velocidade máxima, resposta transitória, trim e diferença entre frenagem e roda livre. Ajuste também o arrasto e a resposta angular; uma única constante de motor não descreve toda a mecânica.
4. Filme deslizamentos livres do disco sem contato, a várias velocidades. Use `python calibrate.py puck ensaio.csv` para ajustar aceleração e obter uma estimativa de atrito de Coulomb. CSV: `time,x,y` em segundos e centímetros. Recorte o trecho de deslizamento antes de parar ou bater.
5. Faça impactos em parede, lateral e pá em vários ângulos. Compare velocidades de entrada/saída, torque e dispersão com a simulação. Use os próprios ensaios como regressões numéricas e registre a faixa válida do modelo.
6. Meça atraso total da exposição até o movimento por um evento visual e resposta dos motores; registre percentis e perdas. O tempo de uma chamada BLE não é RTT físico.
7. Ajuste a randomização às distribuições observadas. Aumentar ruído arbitrário não comprova transferência.

O arquivo de calibração deve guardar dados brutos, parâmetros ajustados, resíduos e ensaios reservados. Só declarar fidelidade depois de comparar esses ensaios reservados com previsões feitas antes da medição. O plano 2D não representa tombamentos, saltos ou saída do disco por cima da parede.
