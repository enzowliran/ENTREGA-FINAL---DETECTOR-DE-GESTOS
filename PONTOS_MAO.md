# Parte nova: números pelos pontos da mão (MediaPipe Hand Landmarker)

Este arquivo explica o que foi acrescentado ao projeto em 12/11/2026. Foi escrito para ser lido por quem
não acompanhou a mudança, inclusive por outro assistente de código: dá para colar este arquivo inteiro
numa conversa como contexto.

## Resumo em uma frase

A CNN 3D continua decidindo os gestos de movimento (os 27 do Jester), e os números feitos com a mão
(um, dois, três e quatro dedos e punho) passam a ser decididos por regras sobre os 21 pontos da mão
que o [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
encontra em cada quadro.

## Nada do que já existia foi alterado

Tudo é **arquivo novo**. Não mudou nenhum arquivo existente: nem os `.py`, nem os `.bat`, nem as
páginas HTML, nem os modelos, nem o README, nem o `requirements.txt`, nem o `.gitignore`.
Os atalhos antigos (`abrir_camera.bat`, `abrir_painel.bat` etc.) funcionam exatamente como antes,
com o `cnn_3d_mais_gestos`. A parte nova tem atalhos próprios, terminados em `_pontos`.

## Por que

O `cnn_3d_mais_gestos` aprendeu os números com fotos do HaGRID repetidas como se fossem vídeo. Nas
fotos ele acerta 92,7%, mas fora delas quase não acerta: no Slovo, "um dedo" ficou em primeiro lugar
em 5% a 10% dos vídeos (`resultados/dedos/diagnostico_escala.json`), e no teste de 09/11 na webcam só
"um dedo" passava do limiar. Contar dedos é um problema de **forma da mão**, não de movimento, e os
pontos do MediaPipe descrevem exatamente a forma, sem depender de fundo, luz ou distância.

Além disso, sem os números a CNN pode voltar a ser o `cnn_3d` (só Jester), que é melhor nas duas
medidas do README: 93,3% contra 92,7% de acerto na validação do Jester, e 0,03 contra 0,06 falsos
disparos por minuto.

A mão aberta continua com a CNN (o "Sinal de pare" do Jester, que já funciona bem).

## Como funciona

```
webcam ──┬── CNN 3D cnn_3d (thread própria, 12 quadros/s) ──→ gatilho de sempre ──┐
         │                                                                       ├──→ disparo
         └── Hand Landmarker (todo quadro, ~20 ms em CPU) ──→ DecisorNumeros ───┘
```

1. **Contagem de dedos** (`src/pontos_mao.py`, `contar_dedos`): usa os pontos 3D em metros
   ("world landmarks"), que não mudam com a distância da câmera nem com a mão inclinada.
   - Indicador a mínimo: o dedo está esticado se estiver reto (distância base→ponta dividida pelo
     comprimento do dedo ≥ `reta`) e apontando para fora da palma (ângulo entre pulso→base e
     base→ponta ≤ `angulo`).
   - Polegar: reto (`polegar_reta`) e afastado do mínimo (distância da ponta do polegar à base do
     mínimo, dividida pela largura da palma, ≥ `polegar_fora`).
2. **Pose** (`classificar_pose`): conta quantos dedos estão esticados, **qualquer combinação**. Por
   exemplo, indicador e médio dão 2, e polegar, indicador e médio dão 3. Há duas exceções, que ficam
   com a CNN: o polegar sozinho (é o joinha ou o polegar para baixo) e os cinco dedos (é a mão aberta).
3. **Quando um número dispara** (`DecisorNumeros`):
   - a mesma pose aparece por 0,5 s (`--tempo-estavel`), pela maioria dos últimos 5 quadros;
   - a mão está parada: o pulso não andou mais que 0,5 palma (`--movimento`) e a mão não aproximou
     nem afastou (`--escala`);
   - para repetir o mesmo número, a pose precisa mudar ou a mão sair antes.
4. **Quem vale em cada momento**: o sistema não precisa saber de antemão se o gesto vai ser parado ou
   com movimento. Os dois lados rodam o tempo todo:
   - **Mão parada numa pose de número:** a CNN fica barrada. Isso evita, por exemplo, que "quatro dedos"
     parado vire "Sinal de pare". Os disparos barrados ficam contados.
   - **Mão em movimento:** os números não disparam, e quem decide é a CNN. É por isso que deslizar com
     dois dedos não dispara "dois".
   - **Depois de qualquer disparo:** o outro lado fica bloqueado por 1 s (`--bloqueio`).

## Arquivos novos

| Arquivo | O que é |
|---|---|
| `src/pontos_mao.py` | Núcleo: detector de mão, contagem de dedos, `DecisorNumeros`, desenho do esqueleto |
| `src/avaliar_pontos_hagrid.py` | Mede a contagem em fotos do HaGRID e ajusta os limiares (metade das pessoas para ajustar, outra metade para medir) |
| `demo/motor_pontos.py` | Junta a CNN (`AnalisadorCNN`, em thread) e os pontos (`SistemaPontos`). Usado pelos três abaixo |
| `demo/ao_vivo_pontos.py` | Janela da câmera com esqueleto da mão, dedos contados e disparos (como o `ao_vivo.py`) |
| `demo/player_pontos.py` | O mesmo servidor das páginas (importa o `player.py` sem alterá-lo). As páginas recebem os mesmos eventos |
| `demo/testar_webcam_pontos.py` | O mesmo teste guiado do `testar_webcam.py` (importado sem alteração), com o sistema novo |
| `demo/python_pontos.bat` | Roda o Python do ambiente `.venv_pontos` e cria esse ambiente sozinho na primeira vez |
| `demo/abrir_camera_pontos.bat` | Janela da câmera |
| `demo/abrir_painel_pontos.bat` | Painel do carro |
| `demo/abrir_hospital_pontos.bat` | Hospital |
| `demo/abrir_apresentacao_pontos.bat` | Apresentação. **Não** roda o `gerar_apresentacao.py`: usa o `apresentacao.html` como está |
| `demo/abrir_teste_webcam_pontos.bat` | Teste guiado na webcam (mesmo roteiro e mesmo envio para o GitHub do original) |
| `requirements_pontos.txt` | Pacotes do `.venv_pontos` |
| `modelos_externos/.gitignore` | O `hand_landmarker.task` (cerca de 8 MB, do Google) é baixado sozinho e fica fora do git |
| `resultados/pontos/` | Resultados da medição no HaGRID (só números) |

## Ambiente separado

O MediaPipe instala o `opencv-contrib-python`, que disputa o mesmo `cv2` com o `opencv-python` do
`.venv`. Por isso a parte nova usa outro ambiente, o `.venv_pontos` (fora do git, como o `.venv`):
- **Criação:** na primeira vez que um atalho `_pontos` roda, o `demo/python_pontos.bat` cria o ambiente
  com o [uv](https://docs.astral.sh/uv/). Leva menos de 1 minuto.
- **Conteúdo:** mediapipe 1.2.3, torch 2.15 só CPU, opencv-contrib 5.0 e pillow, todos em Python 3.12.
- **Com GPU NVIDIA:** a parte nova roda a CNN na CPU, porque o torch do `.venv_pontos` é só CPU. Funciona,
  só mais devagar. Para usar a GPU, instale no `.venv_pontos` o torch com CUDA, como diz o
  `requirements.txt` original.

## Como usar

Dê duplo clique nos atalhos ou rode pelo terminal:

```bat
demo\abrir_camera_pontos.bat          :: janela com o esqueleto da mão e os disparos
demo\abrir_painel_pontos.bat          :: painel do carro
demo\abrir_hospital_pontos.bat        :: hospital
demo\abrir_teste_webcam_pontos.bat    :: teste guiado (escolha 2 = só os dedos)
```

Ou, com opções:

```bat
demo\python_pontos.bat demo\ao_vivo_pontos.py --tempo-estavel 0.7 --camera 1
demo\python_pontos.bat src\avaliar_pontos_hagrid.py
```

**Comparar com a CNN sozinha:** as sessões do teste novo vão para a mesma pasta `resultados/webcam/`,
com o modelo anotado como `resultados/jester/cnn_3d/melhor.pt + pontos`. Assim,
`python demo/testar_webcam.py --resumo` mostra a CNN sozinha e a parte nova lado a lado, cada uma
como um "modelo" separado.

## Resultados

### Fotos do HaGRID (`resultados/pontos/hagrid_amostra.json`)

**Amostra e método:**
- 60 fotos de validação por classe, baixadas com
  `python src/baixar_hagrid.py --destino data/HaGRID_amostra --por-classe 0`.
- Cada foto é recortada em volta da mão do gesto.
- As pessoas são divididas em duas metades pelo `user_id`. Os limiares são ajustados numa metade e o
  acerto é medido na outra, a metade de teste, com 410 fotos de 14 classes.
- Para refazer: `demo\python_pontos.bat src\avaliar_pontos_hagrid.py`. Com `--reaproveitar`, ele não
  roda o MediaPipe de novo.

| Medida (metade de teste) | Limiares iniciais | Limiares ajustados (em uso) |
|---|---:|---:|
| MediaPipe encontrou a mão | 94% | 94% |
| Acerto médio nos números (um, dois, três, quatro, punho) | 87,1% | **88,3%** |
| O mesmo, só nas fotos em que achou a mão | 94,0% | **95,0%** |
| Joinha, polegar para baixo e mão aberta sem virar número | 92,4% | 90,5% |

Por classe, com os limiares em uso:

| Classe | Acerto |
|---|---:|
| Quatro dedos | 100% |
| Dois dedos (peace) | 89% |
| Dois dedos (two_up) | 97% |
| Três dedos (three) | 84% |
| Três dedos (three2) | 91% |
| Um dedo | 84% |
| Punho | 74% |

A maior parte dos erros do punho é o MediaPipe não achar a mão fechada (4 de 23 fotos). Quando ele
acha, o punho acerta 17 de 19. Para comparar, a CNN `cnn_3d_mais_gestos` acerta 92,7% nessas mesmas
classes nas fotos, mas cai para 5% a 10% em vídeo real (Slovo). A vantagem esperada dos pontos é
justamente não depender do tipo de imagem. Isso **ainda precisa ser confirmado na webcam**.

**Limiares em uso:** `reta` 0,85, `angulo` 65, `polegar_reta` 0,95 e `polegar_fora` 1,20. A diferença
para os iniciais é pequena, perto do ruído de uma amostra de 30 fotos por classe.

**Riscos que a medição mostrou** (eles serão medidos na webcam pelo bloco "sem gesto"):
- **"ok" vira número:** o círculo de polegar e indicador, com os outros três dedos esticados, conta como
  "quatro" em 23 de 34 fotos.
- **Mão relaxada vira número:** nas fotos sem gesto do HaGRID, cerca de metade vira algum número.
  Na câmera, a exigência de 0,5 s com a mão parada deve barrar boa parte desses casos, mas isso não
  foi medido.

### Testes sem câmera

- Com mãos sintéticas, a contagem acerta as 7 poses: punho, um, dois, três com polegar, quatro, mão
  aberta e joinha.
- Com mãos sintéticas, o decisor se comporta assim:
  - dois dedos parado dispara uma vez, aos 0,5 s;
  - dois dedos deslizando não dispara e não barra a CNN;
  - para repetir, é preciso baixar a mão, ou trocar a pose (por exemplo, de dois para três);
  - a mão aberta parada não dispara número e não barra a CNN.
- Em CPU (`--simular`): o MediaPipe leva cerca de 20 a 30 ms por quadro e a CNN cerca de 170 ms por análise.
- O `player_pontos.py` serve as páginas, e os eventos `/eventos` saem com os mesmos campos de antes,
  mais `pose`, `dedos`, `ms_mao` e, nos comandos, `origem` (`"cnn"` ou `"pontos"`).

**Ainda não foi testado com câmera de verdade.** Com o MediaPipe no mesmo laço, a câmera cai para
perto de 16 a 20 quadros por segundo. A CNN só usa 12 por segundo, então isso não deve afetá-la, mas
convém conferir.

## O que ainda falta

- **Medir na webcam**, com várias pessoas e com o `abrir_teste_webcam_pontos.bat`:
  - o conjunto "dedos", para o acerto dos números;
  - o conjunto "demos", para ver se a CNN barrada não atrapalhou os outros gestos;
  - o bloco "sem gesto", para os falsos disparos por minuto.
- **Ajustar `--tempo-estavel`**, que hoje é 0,5 s: se o começo de um deslize virar número, aumente;
  se os números ficarem lentos, diminua.
- **Só se as regras não bastarem:** treinar uma rede pequena sobre os pontos, com as fotos do HaGRID
  e as gravações de `data/webcam/`.
- **Também pode ser testado depois:** usar os pontos para conferir a CNN, por exemplo cancelar um
  disparo quando não há mão na imagem, ou conferir a direção do pulso nos deslizes.
