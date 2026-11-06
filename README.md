# Controle do carro por gestos, sem tirar os olhos da estrada

Trabalho final de Deep Learning (Insper), tema mobilidade e segurança.

- **Apresentação intermediária:** 12/11/2026
- **Entrega final (notebook, paper e apresentação):** 17-18/11/2026
- **Restrição da disciplina:** a rede é treinada do zero, sem pesos de terceiros. Usar o nosso próprio pré-treino é permitido.

**Pergunta do trabalho:** um comando por gesto só é viável se quase nunca disparar sozinho, por isso a métrica principal é **falsos disparos por minuto**, e não a acurácia. Também queremos saber quanto um pré-treino próprio ajuda quando há poucos dados do contexto do carro.

## Começo rápido: testar sem baixar os datasets

O modelo já treinado está no repositório, então as demos funcionam logo depois de clonar. Precisa de Python 3.12 e webcam. O clone ocupa cerca de 800 MB, porque inclui os pesos de todos os treinos.

```bash
git clone https://github.com/enzowliran/ENTREGA-FINAL---DETECTOR-DE-GESTOS.git
cd ENTREGA-FINAL---DETECTOR-DE-GESTOS
python -m venv .venv
.venv\Scripts\activate

# Com GPU NVIDIA (a versão que usamos, CUDA 13.0). Sem GPU, pule esta linha.
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

Depois, dê dois cliques em um arquivo de `demo/` (no Windows, com a pasta `.venv` criada como acima):

| Arquivo | O que abre | Endereço |
|---|---|---|
| `abrir_painel.bat` | Painel do carro: música, clima, telefone, limpador e faróis | http://127.0.0.1:8770/ |
| `abrir_tenis.bat` | Jogo de tênis | http://127.0.0.1:8769/ |
| `abrir_apresentacao.bat` | Apresentação controlada por gestos | http://127.0.0.1:8767/ |
| `abrir_player.bat` | Player de música simples | http://127.0.0.1:8765/ |
| `abrir_camera.bat` | Janela da webcam com o gesto reconhecido | |

Sem GPU, rode o comando do `.bat` com `--cpu` no final. Para testar sem webcam, use `--simular`:

```bash
python demo/player.py --pagina painel_carro.html --porta 8770 --painel --simular
```

Com `--simular`, a rota `/testar?c=proxima` envia um comando sem precisar da câmera.

### Painel do carro

Clique em **LIGAR** para liberar o som. Ele toca os 5 MP3 de `demo/musicas/`. Para usar outras músicas, coloque os arquivos nessa pasta; se ela estiver vazia, o painel usa músicas sintetizadas.

A voz diz o nome da página a cada troca, e **empurrar a mão** faz ela ler os comandos da tela. A voz em português usa o Chrome e precisa de internet; sem ela, o painel mostra um aviso e continua funcionando sem voz.

| Gesto | Na tela de música |
|---|---|
| Deslizar para a direita | Próxima |
| Deslizar para a esquerda | Última tocada |
| Sinal de pare | Toca ou pausa |
| Polegar para cima ou baixo | Volume |
| Tamborilar os dedos (com a música pausada) | Abre o menu |
| Puxar a mão | Troca de função |
| Empurrar a mão | Explica os comandos |

Os gestos mudam em cada tela, e a legenda à direita mostra os da tela atual. Pelo teclado: setas, espaço, `M` (tamborilar), `R` (função), `E` (explicação) e `F` (tela cheia).

## Novos treinos

Os datasets não estão no repositório, e cada pessoa precisa baixá-los. O Jester ocupa cerca de 27 GB. Na GPU que usamos (RTX 5060 Laptop, 8 GB), cada época do treino completo levou cerca de 12 minutos. O treino exige GPU NVIDIA.

Baixe os vídeos e os rótulos. O script só baixa os vídeos, e sem os rótulos nenhum treino roda:

```bash
python src/baixar_jester.py --destino data/Jester
mkdir data\Jester\rotulos
curl -L -o data/Jester/rotulos/jester-labels.bin https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Jester/jester-labels
tar -xf data/Jester/rotulos/jester-labels.bin -C data/Jester/rotulos
```

Os rótulos devem ficar em `data/Jester/rotulos/jester_labels/`, com `train.csv`, `validation.csv` e os demais.

Para repetir o melhor treino, em uma pasta nova (`--nome`):

```bash
python src/treinar_jester.py --modelo cnn_3d --epocas 20 --lr 1e-3 --peso-falso 1.0 --aug-tempo --aug-espaco --suavizacao 0.05 --frac-sinteticos 0.1 --nome meu_treino
```

**Use sempre um `--nome` novo.** Sem ele, o resultado grava por cima de `resultados/jester/cnn_3d/`, que é o modelo das demos. Para continuar um treino, use `--retomar` com o `--nome` da pasta: ele parte do `ultimo.pt` que está nela e grava por cima dessa pasta. As arquiteturas disponíveis são `cnn_3d`, `cnn_gru` e `cnn_transformer`. Para um teste rápido, use `--limite-treino 2000` e poucas épocas.

Para medir o modelo salvo, troque o caminho pelo do seu treino:

```bash
python src/calibrar_fluxo.py --modelo resultados/jester/meu_treino/melhor.pt   # falsos disparos por minuto
python src/avaliar_roc.py --modelo resultados/jester/meu_treino/melhor.pt      # ROC e AUC
```

O ranking de gestos do modelo atual roda sem o dataset, com `python src/ranking_gestos.py`, porque usa as confianças salvas em `data/interno/janelas_validacao_cnn_3d.npz`. Para outro modelo, gere esse arquivo antes com `src/calibrar_por_gesto.py`.

Para o DriverMHG, a preparação usa o `tar.exe` do Windows:

```bash
python src/baixar_drivermhg.py --destino data/DriverMHG
mkdir data\DriverMHG\anotacao
tar -xf data/DriverMHG/geral.rar -C data/DriverMHG/anotacao
python src/preparar_drivermhg.py --base data/DriverMHG
python src/treinar_drivermhg.py --modelo cnn_3d --mao direita --modalidade ir --inicio resultados/jester/cnn_3d/melhor.pt
```

Troque `--inicio` por `zero` para treinar sem o pré-treino, que é a comparação principal do trabalho.

## Onde estamos

Modelo das demos: `resultados/jester/cnn_3d/melhor.pt`, CNN 3D com 3,52 milhões de parâmetros, treinada do zero. Entrada de 16 frames em 96×96. Todos os números abaixo são do Jester, exceto os da última linha.

| Medida | Valor |
|---|---|
| Acurácia no teste (14.743 vídeos) | 92,6% (top-5: 99,4%) |
| AUC gesto x sem gesto, no teste | 0,9982 |
| Falsos disparos em fluxo simulado, limiar 0,7 e 3 janelas seguidas | 3 em 100,2 min (0,03 por minuto) |
| Gestos detectados nesse fluxo (validação, 12.786 vídeos) | 72,4%, com 5,1% na classe errada |
| Gestos do volante do DriverMHG, com o modelo sem ajuste | 0 de 384 |

O fluxo é simulado com vídeos sem gesto colados em sequência, ou seja, é uma aproximação. O registro completo, com o que deu errado e as decisões, está em `notebook.ipynb`. Parte das células de código desse notebook é anterior às demos atuais, e o código válido é o de `src/` e `demo/`.

| Experimento | Situação |
|---|---|
| Só DriverMHG x Jester seguido de ajuste no DriverMHG (figura principal) | Não feito; o download do DriverMHG está parcial |
| Falsos disparos por minuto e curva de acerto x falso disparo | Feito no Jester |
| RGB x RGB + infravermelho | Não feito |
| Latência | Não medida formalmente |
| Vocabulário de 3, 5 ou 7 gestos | Não feito |
| CNN + GRU x CNN 3D | Não concluído; a GRU foi interrompida |
| CNN + Transformer | Sem conclusão: 15,2% em um treino pequeno, sem ajuste próprio de taxa de aprendizado |

Tentativas já descartadas, para não repetir: limiar por gesto e ajuste fino de esquerda e direita. Os dois não trouxeram ganho real, e os resultados estão no notebook.

## O que não está no repositório

| O que falta | Por quê | Como obter |
|---|---|---|
| Vídeos e rótulos do Jester (26 GB) | Tamanho e licença | Comandos em "Novos treinos" |
| DriverMHG (12 GB baixados, 82 de 125 gravações) | A licença proíbe redistribuir | Comandos em "Novos treinos"; o download recomeça do zero em cada máquina |
| Amostras de frames do DriverMHG (`data/interno/*.png`, `data/DriverMHG/amostra/`) | Limite de frames publicáveis | Geradas a partir do dataset com `scripts_auxiliares/ver_gestos.py` |

Versões que usamos: Python 3.12.14, PyTorch 2.14.1 (CUDA 13.0), NumPy 2.5.4, OpenCV 5.0.0 e scikit-learn 1.9.1.

## Organização

```
demo/          demos, páginas HTML e atalhos .bat
  musicas/     MP3 tocados no painel
  vendor/      bibliotecas usadas pelas páginas
src/           download, preparação, treino e avaliação
resultados/    pesos melhor.pt e ultimo.pt, métricas e logs de cada treino
scripts_auxiliares/  análises avulsas e edições já aplicadas (veja o LEIA-ME)
figuras/       gráficos
notebook.ipynb diário do projeto
notebooks/     exploração dos dados
data/          datasets (fora do git, exceto as confianças em data/interno/)
```

## Como trabalhar juntos

- Crie uma branch para o seu trabalho (`git switch -c treino-gru`) e abra um pull request para a `main`.
- Cada `melhor.pt` tem de 9 a 14 MB, e o GitHub recusa arquivos acima de 100 MB. Envie só os pesos dos treinos que valem a pena.
- Não altere `resultados/jester/cnn_3d/`: as demos e os números acima dependem dele.
- Use seed fixa, grave os resultados em arquivo e nunca escolha configurações olhando o conjunto de teste.

## Licenças: este repositório é público

- **Nunca envie para cá** a pasta `data/` nem frames ou vídeos dos datasets.
- **DriverMHG:** proíbe redistribuir os dados e limita o que pode ser publicado a 10 frames e 2 clipes, salvo autorização da TUM.
- **Jester:** uso em pesquisa, sem frames no paper ou na apresentação.

Citação do DriverMHG: Köpüklü, O., Ledwon, T., Rong, Y., Kose, N., Rigoll, G. *DriverMHG: A Multi-Modal Dataset for Dynamic Recognition of Driver Micro Hand Gestures and a Real-Time Recognition Framework*. arXiv:2003.00951, 2020.
