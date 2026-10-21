# Especificação Técnica de Engenharia Mecânica, CAD Fusion 360 & Dinâmica Física
## Robô de Hockey RoboCore (Classe 1lb / 454 g — HockeyBot / Falcon 3v3)

> **Documento Oficial de Engenharia do Projeto `robocore-hockey-rl`**  
> Modelagem CAD paramétrica para **Autodesk Fusion 360**, lista completa de componentes (BOM), dedução analítica de **Centro de Massa ($CoM$)** e **Tensor de Inércia ($I_{zz}$)** via Teorema de Steiner, dinâmica de contato pneu-piso com transferência dinâmica de carga, e diferenciação tática dos chassis de **Atacante**, **Ala** e **Goleiro**.

---

## 1. Identificação Geral & Restrições de Regulamento

* **Categoria:** Hockey de Robôs 1lb (RoboCore / Falcon).
* **Formato Oficial de Disputa:** **3v3** (3 robôs por equipe simultaneamente, 6 robôs no ringue de $180 \times 120\text{ cm}$).
* **Massa Máxima Permitida:** **$454,0\text{ g}$** ($1,000\text{ lb}$).
* **Envelope Máximo de Gabarito (Bounding Box):**
  * Largura total ($W$ / Eixo $Y$): $\le 17,0\text{ cm}$ ($170\text{ mm}$).
  * Comprimento total ($L$ / Eixo $X$): $\le 16,0\text{ cm}$ ($160\text{ mm}$).
  * Altura máxima ($H$ / Eixo $Z$): $\le 5,2\text{ cm}$ ($52\text{ mm}$).
* **Eletrônica Embarcada Oficial:** Placa **RoboCore Vespa (ESP32-WROOM-32)** com ponte H dupla integrada e rádio Bluetooth BLE 2.4 GHz.
* **Alimentação:** Bateria LiPo 2S (7.4 V) ou 3S (11.1 V) $450\text{ mAh}$ 35C.
* **Propulsão:** Tração diferencial com 2 rodas tratoras em balanço traseiro e patim dianteiro de apoio deslizante (Teflon / UHMW-PE).

---

## 2. Sistema de Coordenadas de Referência do CAD

A origem $(0, 0, 0)$ do modelo paramétrico 3D é posicionada no **centro geométrico da planta do chassi ao nível do piso**:
* **Eixo $+X$ (Longitudinal / Frente):** Aponta na direção do bico/pá frontal côncava (de $x = -8,0\text{ cm}$ na traseira até $x = +8,0\text{ cm}$ na borda frontal).
* **Eixo $+Y$ (Lateral / Direita):** Aponta para o bordo direito do robô (de $y = -8,5\text{ cm}$ na lateral esquerda até $y = +8,5\text{ cm}$ na lateral direita).
* **Eixo $+Z$ (Vertical / Cima):** Aponta verticalmente para cima a partir do plano do solo ($z = 0,0\text{ cm}$).
* **Eixo das Rodas Tratoras ($x_{\text{axle}}$):** $x = -2,5\text{ cm}$ (recuado para favorecer a manobrabilidade e dar vão útil frontal).
* **Vão Livre do Solo (*Ground Clearance*):** $4,0\text{ mm}$ no chassi base, $2,0\text{ mm}$ no patim frontal de contato.
* **Diâmetro das Rodas:** $\varnothing 65,0\text{ mm}$ ($6,5\text{ cm}$) com largura de $16,0\text{ mm}$.
* **Bitola entre Rodas (*Track / Wheelbase*):** $L_w = 14,5\text{ cm}$ ($145\text{ mm}$).

---

## 3. Lista de Materiais Completa (BOM - Bill of Materials) de 454.0 g

Cada grama do robô foi contabilizada e posicionada tridimensionalmente para refletir a montagem física real de bancada:

| Item | Componente / Sub-Montagem | Material / Especificação | Qtd | Massa Un. (g) | Massa Tot. (g) | $CoM_X$ (cm) | $CoM_Y$ (cm) | $CoM_Z$ (cm) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **01** | Chassi Monocoque Estrutural Base | PETG / Nylon PA12-CF ($\rho=1.25\text{ g/cm}^3$) | 1 | 142.0 | **142.0** | $+0.40$ | $0.00$ | $+2.10$ |
| **02** | Tampa Superior de Fechamento | Policarbonato transparente 2mm | 1 | 40.0 | **40.0** | $-0.20$ | $0.00$ | $+4.90$ |
| **03** | Motores DC Micro Metal 25GA/N20 | Cobre / Aço / Latão (redução 30:1) | 2 | 36.5 | **73.0** | $-2.50$ | $\pm 4.20$ | $+2.40$ |
| **04** | Braçadeiras / Suportes dos Motores | Alumínio 6061-T6 usinado | 2 | 6.0 | **12.0** | $-2.50$ | $\pm 5.50$ | $+2.40$ |
| **05** | Rodas de Tração (Cubo + Pneu) | Alumínio 6061 + Pneu Silicone 40A | 2 | 19.0 | **38.0** | $-2.50$ | $\pm 7.25$ | $+2.40$ |
| **06** | Acopladores Sextavados de Latão | Latão C360 usinado (Eixo D $\to$ Roda) | 2 | 6.0 | **12.0** | $-2.50$ | $\pm 6.75$ | $+2.40$ |
| **07** | Placa Controladora RoboCore Vespa | ESP32-WROOM-32 + Ponte H Dupla | 1 | 29.0 | **29.0** | $+0.80$ | $0.00$ | $+3.50$ |
| **08** | Bateria LiPo 2S/3S 450 mAh 35C | Polímero de Lítio (Pack com XT30) | 1 | 44.0 | **44.0** | $+2.40^*$ | $0.00$ | $+1.50$ |
| **09** | Lâmina da Pá Côncava (*Scoop Lip*) | Aço Inox 304 chanfrado (0.8 mm) | 1 | 26.0 | **26.0** | $+6.80$ | $0.00$ | $+0.80$ |
| **10** | Patins Deslizantes de Apoio | Teflon (PTFE) / UHMW-PE ($\mu=0.12$) | 2 | 5.0 | **10.0** | $+6.80 / -6.50$ | $0.00$ | $+0.20$ |
| **11** | Parafusos Allen Inox e Insertos M3 | Aço Inox 304 + Latão recartilhado | 28 | — | **18.0** | $+0.20$ | $0.00$ | $+2.40$ |
| **12** | Chicote, Chave Geral e Conectores | Cabos silicone 20AWG + XT30 + Chave | 1 | 8.0 | **8.0** | $-1.20$ | $+2.20$ | $+2.60$ |
| **13** | Marcadores Visuais de Chassi | LED Proa difuso + Tag Popa | 2 | 1.0 | **2.0** | $\pm 4.50$ | $0.00$ | $+5.00$ |
| **TOTAL** | **Massa Nominal em Ordem de Marcha** | **Pesagem Oficial RoboCore 1lb** | — | — | **454.0 g** | — | — | — |

*\* Nota: A posição longitudinal da bateria ($CoM_X$) varia de $+2,4\text{ cm}$ (Atacante) a $-2,0\text{ cm}$ (Goleiro) para ajustar a distribuição de carga sem alterar o peso total.*

---

## 4. Diferenciação dos 3 Chassis Físicos (Atacante, Ala, Goleiro)

No Hockey de Robôs 3v3 da RoboCore, as funções táticas exigem geometrias frontais e distribuições de massa específicas:

```
[ ATACANTE / STRIKER ]          [ ALA / DEFENSOR ]              [ GOLEIRO / GOALIE ]
    Profundidade: 2.2 cm            Profundidade: 1.8 cm            Profundidade: 0.8 cm
    Largura: 11.0 cm                Largura: 12.0 cm                Largura: 14.0 cm
    ┌───\        /───┐              ┌───\        /───┐              ┌───\            /───┐
    │    \______/    │              │    \______/    │              │    \__________/    │
    │   (Côncavo)    │              │   (Defletores) │              │    (Escudo Reto)   │
    │   CoM = +0.02  │              │   CoM = -0.24  │              │    CoM = -0.37     │
    │   Rodas: 72.9% │              │   Rodas: 75.7% │              │    Rodas: 77.1%    │
    └────────────────┘              └────────────────┘              └────────────────┘
```

### 4.1. Chassi Atacante (Falcon Striker)
* **Perfil da Pá Frontal:** Côncavo parabólico profundo ($2,2\text{ cm}$) com lábio de retenção (*retention lip*) em aço inox chanfrado.
* **Propósito:** Condução do disco em alta velocidade retido no côncavo, sem que o disco escape lateralmente em aceleração retilínea.
* **Ejeção Centrífuga (*Spin Shot*):** Ao rotacionar bruscamente ($\omega > 6,2\text{ rad/s}$), a força centrífuga supera a retenção mecânica e dispara o disco a até $1,8\times$ a velocidade linear do carrinho.
* **Propriedades Físicas Calculadas:**
  * Massa: $454,0\text{ g}$.
  * Centro de Massa: $\bar{x} = +0,024\text{ cm}$, $\bar{y} = +0,039\text{ cm}$, $\bar{z} = 2,384\text{ cm}$.
  * Inércia Rotacional $I_{zz}$: $13.787,2\text{ g}\cdot\text{cm}^2$ ($0,001379\text{ kg}\cdot\text{m}^2$).
  * Carga Normal Estática: **$72,9\%$ nas rodas traseiras** e **$27,1\%$ no patim frontal**.

### 4.2. Chassi Ala / Defensor (Falcon Defender)
* **Perfil da Pá Frontal:** Côncavo intermediário ($1,8\text{ cm}$) com asas defletoras anguladas a $45^\circ$ nas extremidades ($12,0\text{ cm}$ de boca).
* **Propósito:** Afastar o disco das bordas da arena e canalizar rebotes para o meio-campo.
* **Propriedades Físicas Calculadas:**
  * Massa: $454,0\text{ g}$.
  * Centro de Massa: $\bar{x} = -0,237\text{ cm}$, $\bar{y} = +0,039\text{ cm}$, $\bar{z} = 2,390\text{ cm}$.
  * Inércia Rotacional $I_{zz}$: $13.482,6\text{ g}\cdot\text{cm}^2$ ($0,001348\text{ kg}\cdot\text{m}^2$).
  * Carga Normal Estática: **$75,7\%$ nas rodas traseiras** e **$24,3\%$ no patim frontal**.

### 4.3. Chassi Goleiro (Falcon Goalie)
* **Perfil da Pá Frontal:** Escudo frontal amplo ($14,0\text{ cm}$) com rebaixo mínimo ($0,8\text{ cm}$) e vão livre rente ao solo ($1,5\text{ mm}$).
* **Propósito:** Bloqueio de chutes rasteiros em qualquer ângulo da baliza e rebatida imediata sem reter o puck.
* **Propriedades Físicas Calculadas:**
  * Massa: $454,0\text{ g}$.
  * Centro de Massa: $\bar{x} = -0,374\text{ cm}$, $\bar{y} = +0,039\text{ cm}$, $\bar{z} = 2,379\text{ cm}$.
  * Inércia Rotacional $I_{zz}$: $13.672,4\text{ g}\cdot\text{cm}^2$ ($0,001367\text{ kg}\cdot\text{m}^2$).
  * Carga Normal Estática: **$77,1\%$ nas rodas traseiras** e **$22,9\%$ no patim frontal** (máxima aderência lateral para deslizar em frente ao gol).

---

## 5. Dedução Matemática do Centro de Massa & Tensor de Inércia

O cálculo de propriedades inerciais é feito pelo **Teorema dos Eixos Paralelos (Steiner)** somando a inércia local de cada componente $i$ e o produto de sua massa pela distância ao quadrado até o centro de massa baricêntrico:

### 5.1. Posição Baricêntrica
$$\bar{x} = \frac{\sum_{i=1}^{N} m_i \, x_i}{\sum_{i=1}^{N} m_i}, \quad \bar{y} = \frac{\sum_{i=1}^{N} m_i \, y_i}{\sum_{i=1}^{N} m_i}, \quad \bar{z} = \frac{\sum_{i=1}^{N} m_i \, z_i}{\sum_{i=1}^{N} m_i}$$

Com $M = \sum m_i = 454,0\text{ g}$.

### 5.2. Tensor de Inércia Global
$$I_{xx} = \sum_{i=1}^{N} \left[ \frac{m_i}{12} \left( \Delta y_i^2 + \Delta z_i^2 \right) + m_i \left( (y_i - \bar{y})^2 + (z_i - \bar{z})^2 \right) \right]$$

$$I_{yy} = \sum_{i=1}^{N} \left[ \frac{m_i}{12} \left( \Delta x_i^2 + \Delta z_i^2 \right) + m_i \left( (x_i - \bar{x})^2 + (z_i - \bar{z})^2 \right) \right]$$

$$I_{zz} = \sum_{i=1}^{N} \left[ \frac{m_i}{12} \left( \Delta x_i^2 + \Delta y_i^2 \right) + m_i \left( (x_i - \bar{x})^2 + (y_i - \bar{y})^2 \right) \right]$$

Devido à simetria mecânica bilateral em relação ao plano central $Y=0$, os produtos de inércia $I_{xy} \approx 0$ e $I_{yz} \approx 0$.

---

## 6. Dinâmica de Contato & Transferência de Carga no Simulador

No motor físico ([`core/hockey_physics.py`](file:///c:/Users/caihe/Documents/develop/robocore-hockey-rl/core/hockey_physics.py)), o robô é modelado como um corpo rígido com apoio estático e dinâmico:

### 6.1. Distribuição Estática de Carga Normal
Com as rodas em $x_{\text{axle}} = -2,5\text{ cm}$ e o patim em $x_{\text{skid}} = +6,8\text{ cm}$ (distância entre eixos $d = 9,3\text{ cm}$):
$$N_{\text{wheels, static}} = M \cdot g \cdot \frac{x_{\text{skid}} - \bar{x}}{d}$$
$$N_{\text{skid, static}} = M \cdot g - N_{\text{wheels, static}}$$

### 6.2. Transferência Dinâmica de Carga Longitudinal ($\Delta N$)
Sob aceleração longitudinal $a_x = \frac{d v_{\text{lin}}}{dt}$:
$$\Delta N = M \cdot a_x \cdot \frac{\bar{z}}{d}$$
* **Aceleração para frente ($a_x > 0$):** A carga nas rodas tratoras aumenta ($N_{\text{wheels}} = N_{\text{wheels, static}} + \Delta N$), aumentando a tração disponível.
* **Frenagem brusca ($a_x < 0$):** O peso é transferido para o patim dianteiro, aliviando as rodas tratoras e provocando derrapagem (*wheel slip*) se o piloto frear com excesso de força.

### 6.3. Limite de Tração e Arrasto de Deslizamento
A tração máxima das rodas tratoras de silicone é:
$$F_{\text{traction, max}} = \mu_{\text{tire}} \cdot N_{\text{wheels, dinâmico}}, \quad \text{com } \mu_{\text{tire}} \approx 0,85$$

O patim dianteiro de Teflon em contato com o solo gera arrasto resistente ao avanço:
$$F_{\text{skid\_drag}} = \mu_{\text{skid}} \cdot N_{\text{skid, dinâmico}}, \quad \text{com } \mu_{\text{skid}} \approx 0,12$$

Esse arrasto é subtraído diretamente das rodas tratoras, simulando fielmente a resistência que o carrinho real encontra no piso da arena.

---

## 7. Como Importar e Usar os Modelos no Autodesk Fusion 360

O projeto disponibiliza duas formas oficiais de utilizar o CAD no Fusion 360:

### 7.1. Importação Direta dos Arquivos 3D (OBJ / STL)
Os modelos tridimensionais com escala métrica 1:1 e faces poligonais estão prontos em [`cad/models/`](file:///c:/Users/caihe/Documents/develop/robocore-hockey-rl/cad/models/):
* `chassis_atacante.obj` e `chassis_atacante.stl` (Falcon Striker)
* `chassis_ala.obj` e `chassis_ala.stl` (Falcon Defender)
* `chassis_goleiro.obj` e `chassis_goleiro.stl` (Falcon Goalie)

**Passo a passo no Fusion 360:**
1. Abra o Autodesk Fusion 360.
2. Vá em **File $\to$ Open...** ou arraste o arquivo `.obj` / `.stl` diretamente para a área de trabalho.
3. No painel de importação, certifique-se de que a unidade está definida como **Centímetros (cm)** ou **Milímetros (mm)**.
4. Para converter em corpo sólido: clique com o botão direito na malha $\to$ **Mesh to BRep**.

### 7.2. Execução do Script Oficial da API do Fusion 360
O arquivo [`cad/fusion360_build_chassis_script.py`](file:///c:/Users/caihe/Documents/develop/robocore-hockey-rl/cad/fusion360_build_chassis_script.py) constrói os sólidos paramétricos nativos com sketches, extrusões e furações M3:

**Passo a passo:**
1. No Fusion 360, pressione **Shift + S** ou vá em **Utilities $\to$ Scripts and Add-Ins**.
2. Clique no ícone verde de **+** na aba *Scripts*.
3. Navegue até a pasta `cad/` do projeto e selecione `fusion360_build_chassis_script.py`.
4. Selecione o script na lista e clique em **Run**.
5. O chassi paramétrico completo será modelado automaticamente no seu documento com as cotas exatas de regulamento!

### 7.3. Regeneração dos Arquivos via Linha de Comando
Para recalcular propriedades de massa e regenerar todos os arquivos CAD a qualquer momento:

```powershell
python cad/fusion_chassis_generator.py
```

O script reportará a pesagem componente a componente e salvará o arquivo consolidado [`cad/models/chassis_specs.json`](file:///c:/Users/caihe/Documents/develop/robocore-hockey-rl/cad/models/chassis_specs.json).
