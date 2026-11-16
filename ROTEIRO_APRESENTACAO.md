# Roteiro da apresentação final: Gestos ao volante

15 minutos de fala, 30 de perguntas, 3 pessoas. Os tempos abaixo somam 14min35, com 25 segundos de folga.
Abrir com `demo\abrir_apresentacao_final.bat`. Os números citados aqui são os mesmos dos slides.

## Divisão sugerida

| Quem | Slides | Tempo | Assunto |
|---|---|---|---|
| Pessoa 1 | 1 a 4 | 3min40 | O problema e a solução |
| Pessoa 2 | 5 a 8 | 5min45 | Como funciona e as duas demos ao vivo |
| Pessoa 3 | 9 a 14 | 5min10 | Resultados, impacto, limitações e próximos passos |

Quem faz a demo do carro (slide 8) deve ser quem mais treinou os gestos na webcam.

## Pessoa 1: o problema e a solução (3min40)

**1 · Capa (20 s).** Nome do projeto e do grupo. Uma frase: "comandar a central do carro sem tirar os olhos da estrada".

**2 · O problema (1 min).** Os comandos do carro foram para uma tela de toque. Para trocar a música, o motorista precisa olhar, mirar e tocar. São três custos: olhar para a tela, tirar a mão do volante e navegar em menus.

**3 · O problema, em números (1min20).**
- A 110 km/h, tarefas simples levam 10 s com botões e 30,4 s na tela: cerca de 930 m sem olhar a estrada.
- Mexer em tela ou aparelho do carro multiplica o risco de acidente por 4,6 (estudo SHRP2).
- 3.275 mortes nos EUA em 2023 em acidentes com distração.
- A Euro NCAP passa a dar nota menor, a partir de 2026, a carros que dependem da tela.

**4 · A solução (1 min).** Uma câmera no painel reconhece gestos da mão. Gestos simples, feitos sem desviar o olhar. O sinal de OK ativa o sistema por 7 segundos; fora disso, nada é executado. Passar a palavra: "agora, como isso funciona por dentro".

## Pessoa 2: como funciona e as demos (5min45)

**5 · Como funciona (1min30).** Ler o slide da esquerda para a direita, em três passos.
1. *Ativar:* o sinal de OK abre uma janela de 7 s, com bip e voz.
2. *Reconhecer:* a cada instante o sistema faz uma pergunta: a mão está parada, mostrando de 1 a 4 dedos?
   - Se sim, é um número, e quem responde é a **contagem de dedos**: o MediaPipe marca 21 pontos da mão e uma regra de geometria conta os dedos esticados. Não é rede neural.
   - Se não, quem responde é a **CNN 3D**, a rede neural que lê o vídeo e reconhece o movimento.
3. *Executar:* o comando vai para a central, com resposta por voz.

Frase para fechar: "número parado é com a contagem de dedos; movimento é com a CNN".

**6 · O que o sistema vê, ao vivo (1 min).** Mostrar a mão: a imagem da câmera, os 21 pontos e a mão em 3D. Abrir e fechar os dedos para a plateia ver os pontos acompanhando.

**7 · Os gestos (45 s).** São 11 gestos. Três números (1, 2 e 4 dedos) abrem Música, Ar-condicionado e Telefone. Oito movimentos fazem o resto: deslizar troca a música, polegar muda o volume, pare toca ou pausa. Não ler a tabela inteira.

**8 · Demo do carro, ao vivo (2min30).** Sequência curta sugerida (ensaiar antes):
1. Sinal de OK (a janela abre). Um dedo: abre Música.
2. OK. Deslizar para a direita: próxima música. Polegar para cima: volume.
3. OK. Dois dedos: abre Ar-condicionado. Sinal de pare: liga ou desliga.
4. OK. Quatro dedos: abre Telefone.

Dicas: mão parada por meio segundo nos números; soltar o OK antes de fazer o gesto; se um gesto falhar, repetir uma vez e seguir.

## Pessoa 3: resultados, impacto e próximos passos (5min10)

**9 · Resultados: acerto (1min15).**
- O modelo final acerta **87,7%** dos 11 gestos do carro, em 1.300 clipes de pessoas que não estavam no treino.
- Movimentos, pela CNN 3D: 91,2%. Números, pela contagem de dedos: 82,0%.
- Por que a contagem: a CNN sozinha acertava só 58,4% dos números.
- Se perguntarem do 70,9%: é o acerto nas 36 classes que a CNN conhece; 25 delas não são comando no carro.

**10 · Resultados: disparos sozinhos (1min15).**
- Em 31,5 min de vídeo real sem gesto, e sem a janela do OK, o modelo dispara 1,05 comando do carro por minuto.
- Todos os 33 falsos são números: uma mão parada à toa parece um número.
- Por isso existe a ativação pelo OK: fora dos 7 s, a mão parada não dispara nada.
- Ser honesto: a taxa com a janela incluída ainda não foi medida.

**11 · Impacto (50 s).** Com a tela, 30,4 s sem olhar a estrada; com botões, 10 s; com o gesto, o motorista continua olhando para a frente. E o mercado está indo nessa direção (Euro NCAP).

**12 · Limitações (50 s).** Escolher três para falar: câmera comum não funciona à noite; ainda dispara sozinho sem a janela do OK; os modelos com mais acerto nos dados não reconheceram os números na webcam.

**13 · Se tivéssemos mais tempo (50 s).** Dois passos: gravar vídeos da nossa webcam para treinar a GRU e medir no cenário real; testar numa placa FPGA com câmera infravermelha.

**14 · Obrigado (10 s).** Abrir para perguntas.

## Perguntas prováveis e onde está a resposta

| Pergunta | Resposta curta | Slide |
|---|---|---|
| Por que não usaram o modelo com maior acerto? | Três versões foram melhores nos dados (92,2%, 92,0% e 87,9%). Na webcam, nenhuma reconhecia os números. No teste de 16/11, o veto da CNN cancelaria 116 de 116 leituras de número. | A8 · Por que não o de maior acerto |
| O que é o veto que vocês desligaram? | A CNN podia cancelar um número quando achava que via outro gesto. Ajudava nos dados e derrubava os números na webcam. | A8 e A7 · Contagem de dedos |
| Como a contagem de dedos funciona? | Dedo esticado: base até a ponta ÷ comprimento do dedo ≥ 0,85 e ângulo ≤ 65°. Mão parada e mesma contagem em 5 de 6 quadros. | A7 · Como contamos os dedos |
| Quem decide entre a CNN e a contagem? | A contagem tem prioridade nos números; em todo o resto, a CNN. A CNN nunca responde número. | A7 · O modelo final |
| Como é a CNN 3D? | 4 blocos de convoluções 3×3×3 (32 a 256 filtros), 3,52 milhões de parâmetros, vídeo de 16 quadros de 96×96. | A4 · Arquitetura |
| Como ela foi treinada? | Do zero no Jester (92,6% no teste); depois ajustada nas cinco bases, com 36 classes (84,0% no teste misto). | A4 e A8 · Do primeiro modelo ao final |
| Que outras arquiteturas testaram? | Seis, no mesmo protocolo: CNN 3D, larga, leve, ResNet 3D, CNN 2D + GRU e CNN 2D + Transformer. | A8 · Arquiteturas do zero |
| E a GRU? | Rede pequena sobre os 21 pontos. Boa nos dados; ao vivo, nota baixa nos números. Não é usada. | A6 |
| Por que os dados e a webcam divergem? | Causa provável: outra câmera, distância e fundo; a GRU aprendeu os números com fotos. | A8 |
| Como mediram os falsos disparos? | Vídeo contínuo do IPN, 13 pessoas fora do treino, mesmo laço das demos. | A11 |
| Que dados usaram? | Cinco bases, sempre separando as pessoas de treino e de teste. | A1 |
| Por que não só a voz? | A voz também atrasa a reação e falha com ruído, conversa e ligação. | A2 |
| Como iria para um carro? | Contagem de dedos na CPU; CNN no acelerador de IA; câmera infravermelha. | A12 |

## O que não dizer

- Não dizer que o modelo final tem 92%: esse número é de uma versão que não funcionou na webcam.
- Não dizer que os falsos disparos com a janela do OK foram medidos: ainda não foram.
- O teste ao vivo de 16/11 foi com um usuário, por 2 minutos, sem anotar gesto a gesto: serve para mostrar o efeito do veto, não é uma taxa de acerto.
