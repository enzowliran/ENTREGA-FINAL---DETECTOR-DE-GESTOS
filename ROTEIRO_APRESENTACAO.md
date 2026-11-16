# Roteiro da apresentação final: Gestos ao volante

15 minutos de fala, 30 de perguntas, 3 pessoas. São 13 slides no corpo e 20 no apêndice.
Os tempos abaixo somam 14 minutos, com 1 minuto de folga para as demos.
Abrir com `demo\abrir_apresentacao_final.bat`. Os números citados aqui são os mesmos dos slides.

## Divisão sugerida

| Quem | Slides | Tempo | Assunto |
|---|---|---|---|
| Pessoa 1 | 1 a 5 | 5min10 | O problema, a solução e como funciona |
| Pessoa 2 | 6 a 9 | 4min50 | As duas demos ao vivo, os gestos e a ativação |
| Pessoa 3 | 10 a 13 | 4min | Resultados, impacto e próximos passos |

Quem faz a demo do carro (slide 9) deve ser quem mais treinou os gestos na webcam.

## Pessoa 1: o problema, a solução e como funciona (5min10)

**1 · Capa (20 s).** Nome do projeto e do grupo. Uma frase: "comandar a central do carro sem tirar os olhos da estrada".

**2 · O problema (1 min).** Os comandos do carro foram para uma tela de toque. Para trocar a música, o motorista precisa olhar, mirar e tocar. São três custos: olhar para a tela, tirar a mão do volante e navegar em menus.

**3 · O problema, em números (1min20).**
- A 110 km/h, tarefas simples levam 10 s com botões e 30,4 s na tela: cerca de 930 m sem olhar a estrada.
- Mexer em tela ou aparelho do carro multiplica o risco de acidente por 4,6 (estudo SHRP2).
- 3.275 mortes nos EUA em 2023 em acidentes com distração.
- A Euro NCAP passa a dar nota menor, a partir de 2026, a carros que dependem da tela.

**4 · A solução (1 min).** Uma câmera no painel reconhece gestos da mão. Gestos simples, feitos sem desviar o olhar. O sinal de OK ativa o sistema por 7 segundos; fora disso, nada é executado.

**5 · Como funciona (1min30).** Ler o slide da esquerda para a direita, em três passos.
1. *Ativar:* o sinal de OK abre uma janela de 7 s, com bip e voz.
2. *Reconhecer:* a cada instante o sistema faz uma pergunta: a mão está parada e mostrando de 1 a 4 dedos?
   - Se sim, é um número, e quem responde é a **contagem de dedos**: o MediaPipe marca 21 pontos da mão e uma regra de geometria conta os dedos esticados. Não é rede neural.
   - Se não, quem responde é a **CNN 3D**, a rede neural que lê o vídeo e reconhece o movimento.
3. *Executar:* o comando vai para a central, com resposta por voz.

Frase para fechar: "número parado é com a contagem de dedos; movimento é com a CNN".

## Pessoa 2: as demos, os gestos e a ativação (4min50)

**6 · O que o sistema vê, ao vivo (50 s).** Mostrar a mão: a imagem da câmera, os 21 pontos e a mão em 3D. Abrir e fechar os dedos para a plateia ver os pontos acompanhando.

**7 · Os gestos (40 s).** São 11 gestos. Três números (1, 2 e 4 dedos) abrem Música, Ar-condicionado e Telefone. Oito movimentos fazem o resto: deslizar troca a música, polegar muda o volume, pare toca ou pausa. Não ler a tabela inteira.

**8 · A ativação pelo OK (50 s).**
- Um comando errado também distrai o motorista. Por isso nada vale antes do sinal de OK.
- Sem a ativação, em vídeo real sem gesto, o sistema dispara 1,05 comando falso por minuto: uma mão parada à toa parece um número.
- Com a ativação, na simulação com os mesmos vídeos, cai para 0,73. Ainda é alto: o detector do OK abre sozinho 3 vezes por minuto nesses vídeos, e deixá-lo mais exigente é um dos próximos passos.

**9 · Demo do carro, ao vivo (2min30).** Sequência curta sugerida (ensaiar antes):
1. Sinal de OK (a janela abre). Um dedo: abre Música.
2. OK. Deslizar para a direita: próxima música. Polegar para cima: volume.
3. OK. Dois dedos: abre Ar-condicionado. Sinal de pare: liga ou desliga.
4. OK. Quatro dedos: abre Telefone.

Dicas: mão parada por meio segundo nos números; soltar o OK antes de fazer o gesto; se um gesto falhar, repetir uma vez e seguir.

## Pessoa 3: resultados, impacto e próximos passos (4 min)

**10 · Resultados (1min30).**
- O modelo final acerta **87,7%** dos 11 gestos do carro, em 1.300 clipes de pessoas que não estavam no treino.
- Movimentos, pela CNN 3D: 91,2%. Números, pela contagem de dedos: 82,0%.
- Por que a contagem: a CNN sozinha acertava só 58,4% dos números.
- Se perguntarem do 70,9%: é o mesmo modelo, medido nas 36 classes que a CNN conhece; 25 delas não são comando no carro.

**11 · Impacto (1 min).** Com a tela, 30,4 s sem olhar a estrada; com botões, 10 s; com o gesto, o motorista continua olhando para a frente. E o mercado está indo nessa direção (Euro NCAP).

**12 · Próximos passos (1min20).** Quatro passos, cada um responde a uma limitação de hoje:
1. Treinar um modelo só com a mão em 3D: rosto, fundo e luz deixam de interferir.
2. Gravar o cenário real, com a nossa câmera e motoristas.
3. Uma ativação mais exigente, e medir os comandos falsos dentro do carro.
4. Câmera infravermelha, para a noite, numa placa embarcada.

**13 · Obrigado (10 s).** Abrir para perguntas.

## As porcentagens, para ninguém se confundir

| Número | O que é |
|---|---|
| 87,7% | Modelo final, quando a pessoa faz um dos 11 gestos do carro. É o número do slide 10. |
| 70,9% | O mesmo modelo final, medido nas 36 classes que a CNN conhece. Cai porque a contagem lê números em gestos de dois dedos que o carro não usa. |
| 91,2% e 82,0% | As duas partes do 87,7%: movimentos (CNN 3D) e números (contagem de dedos). |
| 58,4% | A CNN sozinha nos números do carro. É o motivo de existir a contagem de dedos. |
| 95,6% | GRU + CNN nos gestos do carro. Melhor nos dados; na webcam, não reconhecia os números. |
| 86,2% | Contagem + veto da CNN nos gestos do carro. Na webcam, o veto cancelava os números. |
| 99,6% | A CNN coloca o gesto certo entre as suas 5 maiores notas. O carro só executa a maior, então não é o número principal. |

## Perguntas prováveis e onde está a resposta

| Pergunta | Resposta curta | Slide |
|---|---|---|
| O modelo de vocês é o melhor? | Nos dados, não: a GRU + CNN chegou a 95,6% nos gestos do carro. Na webcam, em uso real, o nosso foi o único que reconheceu os números. | A8 · Por que não o de maior acerto |
| O que é o veto que vocês desligaram? | A CNN podia cancelar um número quando achava que via outro gesto. No teste ao vivo, cancelaria 116 de 116 leituras de número. | A8 |
| Como o sistema acha a mão e sabe que está parada? | O MediaPipe acha a palma e marca 21 pontos. A regra de mão parada é nossa: o pulso andou no máximo meia palma em 0,5 s. | A7 · MediaPipe e mão parada |
| Como a contagem de dedos funciona? | Dedo esticado: base até a ponta ÷ comprimento do dedo ≥ 0,85 e ângulo ≤ 65°. Mesma contagem em 5 de 6 quadros. | A7 · Como contamos os dedos |
| Como é a CNN 3D? | 4 blocos de convoluções 3×3×3 (32 a 256 filtros), 3,52 milhões de parâmetros, vídeo de 16 quadros de 96×96. | A4 e A8 · Arquiteturas do zero |
| Como ela foi treinada? | Do zero no Jester (92,6% no teste); depois ajustada nas cinco bases, com 36 classes (84,0% no teste misto). | A8 · Do primeiro modelo ao final |
| Que outras arquiteturas testaram? | Seis, no mesmo protocolo: CNN 3D, larga, leve, ResNet 3D, CNN 2D + GRU e CNN 2D + Transformer. | A8 · Arquiteturas do zero |
| E a GRU? | Rede pequena sobre os 21 pontos. Boa nos dados; ao vivo, nota baixa nos números. Não é usada. | A6 |
| Por que tantos disparos falsos? | São números lidos com a mão parada à toa. A ativação pelo OK reduz, mas o OK ainda abre sozinho; é um próximo passo. | Slide 8 e A10 |
| Como mediram os falsos disparos? | Vídeo contínuo do IPN, 13 pessoas fora do treino, mesmo laço das demos. | A11 |
| Que dados usaram? | Cinco bases, sempre separando as pessoas de treino e de teste. Jester: 148.092 vídeos. | A1 |
| Por que não só a voz? | A voz também atrasa a reação e falha com ruído, conversa e ligação. | A2 |
| Como iria para um carro? | Contagem de dedos na CPU; CNN no acelerador de IA; câmera infravermelha. | A12 |

## O que não dizer

- Não dizer que o modelo final tem 92% ou 95%: esses números são de versões que não funcionaram na webcam.
- Não dizer que a ativação resolve os falsos: na simulação, reduz de 1,05 para 0,73 por minuto.
- O teste ao vivo de 16/11 foi com um usuário, por 2 minutos, sem anotar gesto a gesto: serve para mostrar o efeito do veto, não é uma taxa de acerto.
