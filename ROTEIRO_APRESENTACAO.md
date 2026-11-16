# Roteiro da apresentação final: Gestos ao volante

15 minutos de fala, 30 de perguntas, 3 pessoas. São 14 slides no corpo e 21 no apêndice.
Os tempos abaixo somam 14 minutos, com 1 minuto de folga para as demos.
Abrir com `demo\abrir_apresentacao_final.bat`. Os números citados aqui são os mesmos dos slides.

## Divisão sugerida

| Quem | Slides | Tempo | Assunto |
|---|---|---|---|
| Pessoa 1 | 1 a 5 | 5min10 | O problema, a solução e como funciona |
| Pessoa 2 | 6 a 9 | 4min45 | As duas demos ao vivo, os gestos e a ativação |
| Pessoa 3 | 10 a 14 | 4min05 | O que construímos, resultados, impacto e próximos passos |

Quem faz a demo do carro (slide 9) deve ser quem mais treinou os gestos na webcam.

## Pessoa 1: o problema, a solução e como funciona (5min10)

**1 · Capa (20 s).** Nome do projeto e do grupo. Uma frase: "comandar a central do carro sem tirar os olhos da estrada".

**2 · O problema (1 min).** Os comandos do carro foram para uma tela de toque. Para trocar a música, o motorista precisa olhar, mirar e tocar. São três custos: olhar para a tela, tirar a mão do volante e navegar em menus.

**3 · O problema, em números (1min20).**
- A 110 km/h, tarefas simples levam 10 s com botões e 30,4 s na tela: cerca de 930 m sem olhar a estrada.
- Mexer em tela ou aparelho do carro multiplica o risco de acidente por 4,6 (estudo SHRP2).
- 3.275 mortes nos EUA em 2023 em acidentes com distração.

Guardar a Euro NCAP para o slide de impacto.

**4 · A solução (1 min).** Uma câmera no painel reconhece gestos da mão. Gestos simples, feitos sem desviar o olhar. Um sinal com a mão liga o sistema; sem ele, nenhum gesto é executado.

**5 · Como funciona (1min30).** Ler o slide da esquerda para a direita, em três passos.
1. *Ativar:* o sinal de OK abre a janela em que os gestos valem. O detalhe vem no slide 8.
2. *Reconhecer:* a cada instante o sistema faz uma pergunta: a mão está parada e mostrando de 1 a 4 dedos?
   - Se sim, é um número, e quem responde é a **contagem de dedos**: o MediaPipe marca 21 pontos da mão e uma regra de geometria conta os dedos esticados.
   - Se não, quem responde é a **CNN 3D**, a rede neural que lê o vídeo e reconhece o movimento.
3. *Executar:* o comando vai para a central, com resposta por voz.

Frase para fechar: "número parado é com a contagem de dedos; movimento é com a CNN".

## Pessoa 2: as demos, os gestos e a ativação (4min45)

**6 · O que o sistema vê, ao vivo (45 s).** Mostrar a mão: a imagem da câmera, os 21 pontos e a mão em 3D. Abrir e fechar os dedos para a plateia ver os pontos acompanhando.

**7 · Os gestos (40 s).** São 11 gestos, mais o OK, que ativa. Três números (1, 2 e 4 dedos) abrem Música, Ar-condicionado e Telefone. Oito movimentos fazem o resto: deslizar troca a música, polegar muda o volume, pare toca ou pausa. Não ler a tabela inteira.

**8 · A ativação pelo OK (50 s).**
- Um comando errado também distrai o motorista. Por isso nada vale antes do sinal de OK: polegar no indicador por 0,4 s abre 7 segundos para o comando, com bip e voz.
- Em 31,5 minutos de vídeo real sem gesto, nenhum movimento foi disparado por engano: a CNN 3D não confunde a mão em repouso com um comando.
- Com a ativação, os números disparados por engano caem 30%: passam a ser precisos dois sinais seguidos, o OK e o gesto.

**9 · Demo do carro, ao vivo (2min30).** Sequência curta sugerida (ensaiar antes):
1. Sinal de OK (a janela abre). Um dedo: abre Música.
2. OK. Deslizar para a direita: próxima música. Polegar para cima: volume.
3. OK. Dois dedos: abre Ar-condicionado. Sinal de pare: liga ou desliga.
4. OK. Quatro dedos: abre Telefone.

Dicas: mão parada por meio segundo nos números; soltar o OK antes de fazer o gesto; se um gesto falhar, repetir uma vez e seguir.

## Pessoa 3: o que construímos, resultados, impacto e próximos passos (4min05)

**10 · O que construímos (1min15).** Quatro etapas, uma frase para cada:
1. Comparamos 6 arquiteturas treinadas do zero e escolhemos a CNN 3D.
2. A CNN 3D chegou a 92,6% no teste do Jester, com 27 gestos de movimento.
3. A contagem de dedos resolveu os números: 23,6 pontos a mais, de 58,4% para 82,0%.
4. As duas partes juntas comandam 11 gestos ao vivo, na webcam.

Não falar o 87,7% aqui: ele é a abertura do slide seguinte.

**11 · Resultados (1min10).**
- O modelo final acerta **87,7%** dos 11 gestos do carro, em 1.300 clipes de pessoas que não estavam no treino.
- Movimentos: 91,2%. Números: 82,0%.
- Destaques do gráfico: quatro dedos 97%, polegar para baixo 98%, polegar para cima e empurrar a mão 95%.
- A CNN coloca o gesto certo entre as 5 maiores notas em 99,6% dos clipes.

**12 · Impacto (40 s).** Retomar o número do começo: 30,4 s sem olhar a estrada com a tela; com o gesto, o motorista continua olhando para a frente. E o mercado vai nessa direção: a partir de 2026, a Euro NCAP dá nota menor a carros que dependem da tela.

**13 · Próximos passos (50 s).** Do protótipo no notebook para dentro do carro:
1. Treinar um modelo só com a mão em 3D: rosto, fundo e luz deixam de interferir.
2. Gravar o cenário real, com a nossa câmera e motoristas.
3. Uma ativação ainda mais exigente, medida dentro do carro.
4. Câmera infravermelha, para a noite, numa placa embarcada.

**14 · Obrigado (10 s).** Abrir para perguntas.

## Perguntas prováveis e onde está a resposta

As respostas abaixo são para o Q&A; não precisam entrar na fala.

| Pergunta | Resposta curta | Slide |
|---|---|---|
| Por que escolheram este modelo? | Testamos versões com mais acerto nos dados (GRU + CNN: 95,6% nos gestos do carro). Na webcam, em uso real, o nosso foi o único que reconheceu os números. | A8 · Por que escolhemos este modelo |
| O que é o veto que vocês testaram? | A CNN podia cancelar um número quando achava que via outro gesto. No teste ao vivo, cancelaria 116 de 116 leituras de número; por isso ficou desligado. | A8 |
| Como o sistema acha a mão e sabe que está parada? | O MediaPipe acha a palma e marca 21 pontos. A regra de mão parada é nossa: o pulso andou no máximo meia palma em 0,5 s. | A7 · MediaPipe e mão parada |
| Como a contagem de dedos funciona? | Dedo esticado: base até a ponta ÷ comprimento do dedo ≥ 0,85 e ângulo ≤ 65°. Mesma contagem em 5 de 6 quadros. | A7 · Como contamos os dedos |
| Como é a CNN 3D? | 4 blocos de convoluções 3×3×3 (32 a 256 filtros), 3,52 milhões de parâmetros, vídeo de 16 quadros de 96×96. | A4 e A8 · Arquiteturas do zero |
| Como ela foi treinada? | Do zero no Jester (92,6% no teste); depois ajustada nas cinco bases. | A8 · Do primeiro modelo ao final |
| Que outras arquiteturas testaram? | Seis, no mesmo protocolo: CNN 3D, larga, leve, ResNet 3D, CNN 2D + GRU e CNN 2D + Transformer. | A8 · Arquiteturas do zero |
| E a GRU? | Rede pequena sobre os 21 pontos. Boa nos dados; ao vivo, nota baixa nos números. É um dos próximos passos, com vídeos da nossa câmera. | A6 |
| Quantos comandos saem por engano? | Sem a ativação, 1,05 por minuto, todos números; com ela, 0,73 na simulação. Nenhum movimento por engano. Uma ativação mais exigente é o próximo passo. | A11 |
| Como mediram isso? | Vídeo contínuo do IPN, 13 pessoas fora do treino, mesmo laço das demos. | A11 |
| Que dados usaram? | Cinco bases, sempre separando as pessoas de treino e de teste. Jester: 148.092 vídeos. | A1 |
| Por que não só a voz? | A voz também atrasa a reação e falha com ruído, conversa e ligação. | A2 |
| Como iria para um carro? | Contagem de dedos na CPU; CNN no acelerador de IA; câmera infravermelha. | A12 |

## Cuidados

- O número do modelo final é 87,7%. Os 92% e 95% são de versões que não funcionaram na webcam.
- "30% menos números por engano" vem de uma simulação nos vídeos do IPN, não de teste no carro.
- O teste ao vivo de 16/11 foi com um usuário, por 2 minutos: mostra o efeito do veto, não é taxa de acerto.
