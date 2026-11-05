# Controle do carro por gestos, sem tirar os olhos da estrada

Trabalho final da disciplina de Deep Learning (Insper), tema mobilidade e segurança.

- **Apresentação intermediária:** 12/11/2026
- **Entrega final (notebook, paper e apresentação):** 17-18/11/2026
- **Restrição da disciplina:** a rede é treinada do zero, com inicialização aleatória e sem pesos de terceiros.

## Pergunta

Ajustar rádio ou ar-condicionado numa tela tira os olhos da estrada por alguns segundos. Um comando por gesto só é viável se quase nunca disparar sozinho. **A métrica principal é a taxa de ativação acidental, em falsos disparos por minuto, e não a acurácia.**

Queremos saber:

1. É possível treinar do zero um reconhecedor de micro-gestos com poucos falsos disparos?
2. Quanto um pré-treino próprio em gestos genéricos ajuda quando há poucos dados do contexto do carro?

## Dados

| | DriverMHG | Jester |
|---|---|---|
| Origem | TU Munique, em simulador de direção | Qualcomm, gestos diante de webcam |
| Tamanho | 25 pessoas, 5.148 segmentos anotados | 148.092 vídeos, 27 classes |
| Formato | RGB, infravermelho e profundidade, 320×240, 30 fps, em gravações contínuas | RGB, 100 px de altura, 12 fps |
| Download | 19,95 GB em 4 partes RAR | 22,8 GB |
| Divisão | Pessoas 1-18 no treino, 19-25 no teste | A definir |

Os dados não entram no git. A licença do DriverMHG proíbe redistribuir os dados e limita o que pode ser publicado a 10 frames e 2 clipes, salvo autorização da TUM.

## Experimentos

1. Treino só no DriverMHG contra pré-treino no Jester seguido de ajuste no DriverMHG
2. Falsos disparos por minuto em fluxo contínuo e curva de acerto × falso disparo
3. RGB contra RGB + infravermelho, incluindo gravações no escuro
4. Latência entre o fim do gesto e o comando
5. Vocabulário com 3, 5 ou 7 gestos
6. CNN 2D + GRU contra CNN 3D

## Estrutura

```
data/          datasets (fora do git)
notebooks/     01_exploracao, 02_preprocessamento, 03_treino_jester, 04_ajuste_drivermhg, 05_experimentos
src/           funções reutilizáveis
figuras/       gráficos do paper
resultados/    métricas em CSV/JSON, de onde saem as tabelas do paper
demo/          demonstração ao vivo
```

Os experimentos usam seed fixa e gravam os resultados em arquivo.

## Citação do DriverMHG

Köpüklü, O., Ledwon, T., Rong, Y., Kose, N., Rigoll, G. *DriverMHG: A Multi-Modal Dataset for Dynamic Recognition of Driver Micro Hand Gestures and a Real-Time Recognition Framework*. arXiv:2003.00951, 2020.
