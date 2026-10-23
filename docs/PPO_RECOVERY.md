# Recuperação PPO: condução preparada até busca livre

O treino de 22/10 aprovou finalização e condução, mas estagnou ao transferir diretamente para busca livre. Na última avaliação completa desse estágio houve zero gols em cinquenta episódios, contato com o disco em 66% e inatividade em aproximadamente 49% dos passos. Isso não demonstra um defeito único no otimizador: a distribuição inicial mudou bastante e a transferência não garantiu retenção das habilidades.

## Mudanças no otimizador

- Ator e crítico agora têm limites de norma de gradiente independentes (0,5 cada). Um erro grande do crítico deixa de reduzir também o gradiente do ator. Continuam usando Adam e as mesmas redes separadas.
- A entropia da distribuição com `tanh` usa a amostra latente e a identidade estável `2 * (log(2) - z - softplus(-2z))`. Evita inverter uma ação já saturada e perder o gradiente que incentiva a saída da saturação.
- O limite de KL usa a divergência analítica entre as distribuições gaussianas anterior e atual, sobre todo o rollout, após cada época. A transformação `tanh` é bijetiva: a KL teórica é preservada. A ação enviada ao simulador ainda tem o pequeno clamp numérico existente. Essa verificação inclui a última atualização da época, mas não desfaz uma época que ultrapassou a meta.
- `value_clip` limita a atualização das estimativas do crítico em relação aos valores coletados. Usa o maior erro quadrático entre estimativa original e limitada; não recorta o retorno. Zero preserva a perda anterior. O plano de recuperação usa 0,5.
- Rollouts não finitos são rejeitados antes de alterar pesos; perdas e normas dos gradientes também são verificadas. Os logs incluem perda do ator/crítico, normas, KL do lote, épocas executadas, número de passos do otimizador, desvio da política e saturação das ações.

Arquitetura, observação de 63 valores, duas ações de rodas e checkpoint v2 permanecem compatíveis. Checkpoints antigos sem `value_clip` retomam com zero. A inferência determinística dos pesos preservados não depende das mudanças no otimizador.

## Currículo e retenção

`configs/training/modular_recovery.json` começa revalidando os pesos aprovados de condução, preservados em `checkpoints/modular-carry-approved-20261022.pt`. A origem e SHA-256 estão no JSON adjacente. Esse modelo não é apresentado como capaz de busca livre ou 1v1.

A sequência é finalização, condução, busca alinhada a 20–40 cm, posicionamento em arcos de 30°, 60°, 120° e 180°, busca livre original e só depois os três módulos 1v1. `solo_setup` amplia os lados de aproximação e a orientação inicial. Posições que precisariam ser corrigidas pelas paredes são reamostradas para respeitar o arco solicitado. Isso condiciona a distribuição de inicialização; o último módulo `solo_search` continua testando a distribuição original, sem essas alterações. Não há controle heurístico das rodas azuis.

Nos estágios intermediários e na busca livre, 20% dos episódios de coleta revisam finalização ou condução. São trajetórias novas da política atual, não replay de experiências antigas ou imitação. Avaliações e GIFs permanecem exclusivos do cenário principal; os episódios do log identificam o cenário efetivamente usado. Sucessos na revisão não reiniciam o contador de estagnação da busca. A revisão é desativada no 1v1.

O plano mantém os pesos do crítico e a exploração entre transferências, com novo otimizador em cada módulo. Usa LR 0,0001, GAE lambda 0,98, rollout 512 × 8 ambientes, entropia 0,005 e KL alvo 0,015. A mudança de lambda amplia a propagação temporal das vantagens; não aumenta o horizonte do episódio.

A recompensa de gol e os demais pesos de fundação são mantidos. A penalidade de excesso de velocidade relativa é desligada nas tarefas de gol: o critério de aproximação lenta pertence à tarefa de chegada, e não deve punir automaticamente a recuperação de um disco em movimento. A dinâmica dos motores e as colisões não mudam. Não se afirma superioridade estatística dessas escolhas antes dos resultados do novo treino.

As regras de aprovação continuam: pelo menos 80% de sucesso ativo em cinquenta episódios de validação e outros cinquenta de confirmação; tarefas de gol anteriores retidas precisam passar vinte episódios cada. O piso não é reduzido para avançar. Estagnação, platô e teto por estágio continuam ativos. As provas repetidas são validação de desenvolvimento, não um teste final independente.

## Rodar

```powershell
.\.venv\Scripts\python.exe -u train_modular.py --plan configs/training/modular_recovery.json --initial checkpoints/modular-carry-approved-20261022.pt --run-dir runs/modular/MINHA_RECUPERACAO --max-hours 4 --keep-awake

.\.venv\Scripts\python.exe train_modular.py --status runs/modular/MINHA_RECUPERACAO
.\.venv\Scripts\python.exe train_modular.py --stop runs/modular/MINHA_RECUPERACAO
```

A execução de 23/10 foi iniciada em `runs/modular/recovery-20261023-025843`, separada das experiências anteriores. `runs/modular-current.json` aponta para a execução atual. `index.html` reúne os GIFs com seed fixa; os JSONs adjacentes têm checkpoint, hash, configuração e resultado. Os pesos e GIFs de execução ficam locais em `runs/`; o checkpoint inicial preservado e as instruções ficam versionados.

Os novos parâmetros também são aceitos em JSON pela UI e CLI. Na CLI comum há `--value-clip`, `--setup-spread`, `--rehearsal-scenario` (repetível) e `--rehearsal-probability`. A UI permite consultar os cenários de revisão e editar sua probabilidade; a lista de cenários é configurada no JSON ou na CLI.

Validação antes do lançamento: 162 testes aprovados, dois ignorados. Os novos testes exercitam estabilidade na saturação, KL incluindo o último passo, independência do gradiente do ator diante de erro alto do crítico, aprendizado de uma ação ótima conhecida, rejeição de dados não finitos, geometria dos estágios, retomada de treino com revisão e isolamento das avaliações.
