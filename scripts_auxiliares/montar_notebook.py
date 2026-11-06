import nbformat as nbf
from pathlib import Path

md = lambda t: nbf.v4.new_markdown_cell(t)
code = lambda t: nbf.v4.new_code_cell(t)


def arquivo(caminho, descricao):
    """Célula que recria o arquivo. Fica fora da execução automática para não reescrever nada."""
    texto = Path(caminho).read_text(encoding="utf-8")
    celula = code(f"%%writefile {caminho}\n{texto}")
    celula.metadata["tags"] = ["skip-execution"]
    celula.metadata["jupyter"] = {"source_hidden": False}
    return [md(f"### `{caminho}`\n\n{descricao}"), celula]


antigo = nbf.read("notebook.ipynb", 4)
diario = antigo.cells[1:]

celulas = [md("""# Controle do carro por gestos, sem tirar os olhos da estrada

Trabalho final da disciplina de Deep Learning (Insper), tema mobilidade e segurança. Este notebook reúne o projeto inteiro: os passos, os comandos, todo o código, os resultados e o diário.

**Tese:** um comando por gesto só é viável se quase nunca disparar sozinho. A métrica principal é a taxa de ativação acidental, e não a acurácia.

**Situação em 04/11/2026:**
- **Pronto:** CNN 3D treinada do zero no Jester, com 92,6% de acerto no teste, além da demo na webcam e do player por gestos.
- **Ainda sem resultado:** o experimento central no volante (DriverMHG), a comparação com e sem pré-treino, os falsos disparos no volante, RGB × infravermelho, latência e tamanho do vocabulário.

**Índice**
1. Como usar este notebook
2. Passo a passo do que foi feito
3. Como reproduzir (comandos)
4. Todo o código
5. Resultados, lidos dos arquivos
6. Coleta de dados no carro
7. Próximos passos
8. Diário

**Licenças:** este notebook não traz frames de nenhum dataset. O DriverMHG limita a publicação a 10 frames e 2 clipes sem aprovação da TUM, e a licença do Jester trata os dados como confidenciais. Cada integrante do grupo deve baixar a própria cópia."""),

md("""## 1. Como usar este notebook

- **Seção 4 (código):** cada célula usa `%%writefile` para recriar um arquivo do projeto. Essas células ficam marcadas com `skip-execution`, mas o botão **Run All** do VS Code ou do Jupyter pode ignorar a marca e sobrescrever os arquivos com este mesmo conteúdo.
- **Seção 5 (resultados):** as células leem os arquivos de `resultados/`, para que nenhum número seja copiado à mão. As células só leem arquivos: não treinam nem baixam nada.
- **Ambiente:** selecione o kernel do `.venv` deste projeto."""),

md("""## 2. Passo a passo do que foi feito

| # | Passo | Resultado |
|---|---|---|
| 1 | Verificar acesso e licença dos datasets | DriverMHG: 19,95 GB, sem cadastro. Jester: 22,93 GB, público, licença só para pesquisa |
| 2 | Criar a estrutura do projeto | Pastas, `README.md` e `.gitignore` |
| 3 | Baixar o DriverMHG sem a profundidade | O RAR não é sólido, então dá para pular a profundidade: cerca de 10 GB em vez de 20 GB. Em andamento: 69 de 125 gravações |
| 4 | Explorar o DriverMHG | 5.148 segmentos, 25 pessoas, cerca de 0,94 s por segmento (`notebooks/01_exploracao.ipynb`) |
| 5 | Baixar o Jester em um arquivo único | 5.279.581 frames em um binário com índice, em vez de milhões de arquivos |
| 6 | Criar as redes do zero | CNN 3D e CNN 2D + GRU e, depois, CNN 2D + transformer |
| 7 | Treino pequeno (20 mil vídeos, 8 épocas) | CNN 3D com 82,8% de acerto na validação de 5 mil vídeos |
| 8 | Testar punição forte de falsa ativação | Peso 5 piorou o modelo: menos gestos no mesmo nível de falsa ativação |
| 9 | Comparar arquiteturas | Transformer com 15,2%, com a mesma taxa de aprendizado. A GRU foi interrompida na 2ª época, então não há conclusão sobre ela |
| 10 | Primeira demo na webcam | Validação de 80%, mas o gesto rápido falhava: a validação não previu esse problema |
| 11 | Corrigir janela e treino | Janelas de 1, 1,5 e 3 s na demo; gestos mais rápidos, mão menor e fora do centro no treino |
| 12 | Treino médio (40 mil vídeos, 10 épocas) | 85,0% na validação; na webcam, reconhecimento bem melhor |
| 13 | Medir falsos disparos em fluxo simulado | 0,02 por min com 29,4% dos gestos detectados no modelo médio |
| 14 | Criar o player por gestos | Esquerda volta, direita avança, sinal de pare alterna play e pause |
| 15 | Treino grande (118.562 vídeos, 20 épocas) | **92,6% no teste**; 0,03 falso por min com 71,8% dos gestos detectados |

**Tela azul:** durante o treino médio, o Windows reiniciou com o erro `0x113`, ligado à NVIDIA. O treino retomou da última época salva. A causa exata não foi determinada.

**Pausa:** o treino grande foi pausado às 18h21 com 8 épocas salvas e retomado às 19h53, sem perda."""),

md("""## 3. Como reproduzir (comandos)

Os comandos rodam na raiz do projeto, no Windows, com o Git Bash.

### Ambiente

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

### Dados

```bash
# Jester: cerca de 35 min a 11 MB/s
.venv/Scripts/python.exe -u src/baixar_jester.py --destino data/Jester
# Os rótulos ficam em data/Jester/rotulos/jester_labels, vindos de:
# https://apigwx-aws.qualcomm.com/qsc/public/v1/api/download/software/dataset/AIDataset/Jester/jester-labels

# DriverMHG sem profundidade: retoma sozinho e a velocidade varia bastante
python -u src/baixar_drivermhg.py --destino data/DriverMHG
.venv/Scripts/python.exe -u src/preparar_drivermhg.py --base data/DriverMHG
```

### Treino no Jester

```bash
# Treino pequeno de validação da ideia (cerca de 15 min)
.venv/Scripts/python.exe -u src/treinar_jester.py --modelo cnn_3d --nome cnn_3d_pequeno --epocas 8 --limite-treino 20000

# Treino médio (cerca de 45 min)
.venv/Scripts/python.exe -u src/treinar_jester.py --modelo cnn_3d --nome cnn_3d_medio --epocas 10 --limite-treino 40000 \\
  --peso-falso 1 --frac-sinteticos 0.1 --aug-tempo --aug-espaco --suavizacao 0.05

# Treino grande (cerca de 4,5 h); acrescente --retomar para continuar da última época salva
.venv/Scripts/python.exe -u src/treinar_jester.py --modelo cnn_3d --nome cnn_3d --epocas 20 \\
  --peso-falso 1 --frac-sinteticos 0.1 --aug-tempo --aug-espaco --suavizacao 0.05
```

### Avaliação

```bash
.venv/Scripts/python.exe -u src/avaliar_jester.py --modelo resultados/jester/cnn_3d/melhor.pt --limite 14787
.venv/Scripts/python.exe -u src/calibrar_fluxo.py --modelo resultados/jester/cnn_3d/melhor.pt
```

### Ajuste no DriverMHG (ainda não rodou de verdade)

```bash
.venv/Scripts/python.exe -u src/treinar_drivermhg.py --modelo cnn_3d --mao direita --modalidade ir --inicio zero
.venv/Scripts/python.exe -u src/treinar_drivermhg.py --modelo cnn_3d --mao direita --modalidade ir --inicio resultados/jester/cnn_3d/melhor.pt
```

### Demos

Dois cliques em `demo/abrir_camera.bat` (tela de teste) ou `demo/abrir_player.bat` (player). As duas usam a mesma câmera, então só uma pode ficar aberta por vez."""),

md("""## 4. Todo o código

Sem pesos pré-treinados de terceiros: as redes começam com inicialização aleatória."""),
]

descricoes = [
    ("src/modelos.py", "As redes: CNN 2D + GRU, CNN 2D + transformer e CNN 3D, e a adaptação do número de canais de entrada."),
    ("src/jester.py", "Carregador do Jester, com decodificação em threads, negativos sintéticos e variações de tempo e espaço."),
    ("src/treinar_jester.py", "Treino no Jester, com a perda de falsa ativação, métricas por limiar e retomada."),
    ("src/avaliar_jester.py", "Mede a falsa ativação por clipe em um modelo salvo e em dados sintéticos."),
    ("src/calibrar_fluxo.py", "Simula um fluxo contínuo e mede falsos disparos por minuto para cada limiar e número de janelas."),
    ("src/baixar_jester.py", "Baixa o Jester como fluxo e grava os frames em um binário com índice."),
    ("src/baixar_drivermhg.py", "Baixa o DriverMHG por trechos, pulando a profundidade, com um RAR por gravação."),
    ("src/preparar_drivermhg.py", "Converte cada gravação em matrizes de acesso rápido."),
    ("src/drivermhg.py", "Leitura das anotações do DriverMHG."),
    ("src/treinar_drivermhg.py", "Treino no DriverMHG, do zero ou a partir do nosso pré-treino. Só foi testado no fluxo, com poucas gravações."),
    ("demo/ao_vivo.py", "Tela de teste na webcam, com legenda, confiança e contador de disparos."),
    ("demo/player.py", "Servidor local do player: roda o modelo na webcam e envia os comandos para a página."),
    ("demo/player.html", "Interface do player, com as 5 músicas geradas no navegador."),
    ("demo/abrir_camera.bat", "Atalho da tela de teste."),
    ("demo/abrir_player.bat", "Atalho do player."),
    ("requirements.txt", "Dependências, além do PyTorch com CUDA."),
    (".gitignore", "Mantém dados e ambiente fora do git."),
]
for caminho, descricao in descricoes:
    celulas += arquivo(caminho, descricao)

celulas += [
md("""## 5. Resultados, lidos dos arquivos

Nada aqui foi digitado à mão: as tabelas e os gráficos vêm de `resultados/jester/`."""),
code("""import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

R = Path("resultados/jester")
pd.set_option("display.width", 160, "display.max_columns", 20)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                     "axes.titlelocation": "left", "axes.titleweight": "bold", "figure.dpi": 110, "legend.frameon": False})
MODELOS = {"cnn_3d_pequeno": "Pequeno", "cnn_3d_pequeno_sint": "Só sintéticos", "cnn_3d_pequeno_p1": "Peso 1",
           "cnn_3d_pequeno_p2": "Peso 2", "cnn_3d_medio": "Médio", "cnn_3d": "Grande"}
CORES = {"Pequeno": "#2a78d6", "Médio": "#eb6834", "Grande": "#1baf7a", "Peso 1": "#eda100", "Só sintéticos": "#e87ba4", "Peso 2": "#6250d6"}"""),
md("### Teste do modelo grande\n\nO conjunto de teste foi usado uma única vez, ao final do treino."),
code("""teste = json.loads((R / "cnn_3d/teste.json").read_text(encoding="utf-8"))
print(f"Teste: top-1 {teste['top1']:.1%}, top-5 {teste['top5']:.1%}, {teste['n']} vídeos")
print(f"Validação usada para escolher a época: {teste['melhor_top1_val']:.1%}")
pd.DataFrame(teste["curva"]).rename(columns={"falsa_ativacao": "falsa ativação", "gestos_acertados": "gestos acertados", "gestos_errados": "gestos errados"}).style.format(
    {"limiar": "{:.2f}", "falsa ativação": "{:.2%}", "gestos acertados": "{:.1%}", "gestos errados": "{:.1%}"}).hide(axis="index")"""),
md("### Evolução no treino\n\nAs validações dos modelos usam conjuntos de tamanhos diferentes: 5 mil, 10 mil e 14.787 vídeos."),
code("""fig, eixo = plt.subplots(figsize=(8.5, 4.2))
for pasta, nome in {"cnn_3d_pequeno": "Pequeno", "cnn_3d_medio": "Médio", "cnn_3d": "Grande"}.items():
    h = pd.read_csv(R / pasta / "historico.csv")
    eixo.plot(h.epoca + 1, h.top1_val * 100, color=CORES[nome], linewidth=2, marker="o", markersize=5, label=nome)
eixo.set_xlabel("Época"); eixo.set_ylabel("Acerto na validação (%)"); eixo.set_ylim(0, 100)
eixo.set_title("Mais dados e mais épocas levaram o acerto a 93%")
eixo.legend(loc="lower right"); fig.tight_layout(); plt.show()"""),
md("### Comparação na validação completa\n\nMesmos 14.787 vídeos, sendo 2.001 sem gesto. Os modelos grande e médio não foram treinados nas mesmas condições dos pequenos."),
code("""linhas, curvas = [], {}
for pasta, nome in MODELOS.items():
    a = json.loads((R / pasta / "avaliacao_falsas.json").read_text(encoding="utf-8"))
    curvas[nome] = pd.DataFrame(a["curva"])
    p = curvas[nome].set_index("limiar")
    linhas.append({"modelo": nome, "acerto": a["top1"], "falsa ativação (0,8)": p.loc[0.8, "falsa_ativacao"], "gestos (0,8)": p.loc[0.8, "gestos_acertados"],
                   "falsa ativação (0,9)": p.loc[0.9, "falsa_ativacao"], "gestos (0,9)": p.loc[0.9, "gestos_acertados"], "ruído (0,6)": a["sinteticos"]["ruido"]["0.6"]})
pd.DataFrame(linhas).style.format({c: "{:.1%}" for c in ["acerto", "gestos (0,8)", "gestos (0,9)"]} | {c: "{:.2%}" for c in ["falsa ativação (0,8)", "falsa ativação (0,9)", "ruído (0,6)"]}).hide(axis="index")"""),
code("""fig, eixo = plt.subplots(figsize=(8.5, 4.4))
for nome in ["Pequeno", "Peso 1", "Médio", "Grande"]:
    c = curvas[nome].sort_values("limiar", ascending=False)
    eixo.plot(c.falsa_ativacao * 100, c.gestos_acertados * 100, color=CORES[nome], linewidth=2, marker="o", markersize=6, label=nome)
eixo.set_xlabel("Falsa ativação por clipe sem gesto (%)"); eixo.set_ylabel("Gestos acertados (%)")
eixo.set_title("O modelo grande acerta mais gestos com pouca falsa ativação")
eixo.legend(loc="lower right"); fig.tight_layout(); plt.show()"""),
md("### Falsos disparos por minuto\n\nFluxo simulado com cerca de 100 min de vídeos sem gesto, colados, e 1.500 vídeos com gesto. Com tão poucos eventos a incerteza é grande, e o valor não vale para o carro."),
code("""todas = []
for pasta, nome in {"cnn_3d_pequeno": "Pequeno", "cnn_3d_pequeno_p1": "Peso 1", "cnn_3d_medio": "Médio", "cnn_3d": "Grande"}.items():
    todas.append(pd.read_csv(R / pasta / "calibracao_fluxo.csv").assign(modelo=nome))
fluxo = pd.concat(todas)
META = 1 / 30
fig, eixo = plt.subplots(figsize=(8.5, 4.4))
for nome in ["Pequeno", "Peso 1", "Médio", "Grande"]:
    f = fluxo[fluxo.modelo == nome]
    eixo.scatter(f.falsos_por_min, f.gestos_detectados * 100, color=CORES[nome], s=36, label=nome, alpha=0.85)
eixo.axvline(META, color="#52514e", linestyle="--", linewidth=1)
eixo.text(META * 1.08, 4, "meta: 1 a cada 30 min", color="#52514e", fontsize=9)
eixo.set_xscale("symlog", linthresh=0.01); eixo.set_xlim(-0.002, 3)
eixo.set_xlabel("Falsos disparos por minuto"); eixo.set_ylabel("Gestos detectados (%)")
eixo.set_title("Dentro da meta, o modelo grande detecta bem mais gestos")
eixo.legend(loc="lower right"); fig.tight_layout(); plt.show()
melhores = fluxo[fluxo.falsos_por_min <= META].sort_values("gestos_detectados", ascending=False).groupby("modelo").head(1)
melhores[["modelo", "limiar", "janelas_seguidas", "falsos_por_min", "falsos", "gestos_detectados", "disparos_classe_errada"]].style.format(
    {"limiar": "{:.2f}", "falsos_por_min": "{:.3f}", "gestos_detectados": "{:.1%}", "disparos_classe_errada": "{:.1%}"}).hide(axis="index")"""),
md("""### O que os resultados permitem concluir

- **A rede aprende do zero:** 92,6% no teste do Jester.
- **O que mais ajudou:** a combinação de mais dados, mais épocas e variações de treino. Como mudou tudo junto, não dá para separar o efeito de cada item.
- **Punição:** nos modelos pequenos, ela não superou o modelo sem punição no mesmo nível de falsa ativação. O peso 1 ficou na receita final, mas isso não prova que ele tenha causado a melhora.
- **Transformer e GRU:** não há conclusão. O transformer usou a mesma taxa de aprendizado, sem ajuste próprio, e a GRU foi interrompida.
- **Validação engana:** com cerca de 80% na validação, a demo ainda falhava com gestos rápidos.
- **Limite principal:** tudo isso vale para mão no ar, diante de webcam. No volante, ainda não há resultado."""),
md("""## 6. Coleta de dados no carro

O prompt abaixo gera, em outra IA, uma página para gravar gestos no carro. **Grave só com o carro parado.** A câmera fica no painel, atrás do volante e acima dele, olhando para baixo em direção ao motorista. Rótulos de esquerda e direita se referem à mão do motorista, e o vídeo não pode ser espelhado.

Os rótulos marcam a janela em que o gesto foi pedido, não o instante exato do movimento. Se aparecer o rosto, trate como dado pessoal. O termo de consentimento deve ser revisado pelo professor.

```text
Crie um único arquivo HTML, com CSS e JavaScript embutidos e sem servidor, para coletar um dataset de gestos de mão no volante de um carro. Interface em português do Brasil. Nada é enviado pela internet.

Segurança: mostre em destaque "Grave apenas com o carro parado, em local seguro. Nunca use dirigindo." e exija marcar "O carro está parado".
Consentimento: antes de ligar a câmera, mostre um termo curto (o que é gravado, uso acadêmico no Insper, direito de desistir e pedir exclusão). Não peça nome nem e-mail.
Configuração: ID do participante (P001), mão dominante, condição de luz, posição da câmera (painel atrás do volante, painel central, outra), lado do volante e câmera.
Vídeo: 1280x720, 30 fps, sem áudio, em WebM com MP4 como alternativa. Pré-visualização e vídeo salvo sem espelhamento.
Gestos, para cada mão: swipe_direita, swipe_esquerda, flick_cima, flick_baixo, tap.
Blocos, cada um em um vídeo contínuo:
1. Gestos: 8 repetições de cada gesto em cada mão, em ordem aleatória com semente salva. Ciclo: Prepare 1,5 s, Agora 2,5 s, Descanse 2 a 4 s. Texto grande e voz (SpeechSynthesis).
2. Gestos rápidos: as mesmas classes, feitos sem olhar e o mais rápido possível, com observacao=rapido.
3. Sem gesto: 2 minutos com as mãos paradas.
4. Outros movimentos: 40 s de cada: ajustar a pegada, girar o volante, coçar o rosto, falar gesticulando, mexer no painel e no câmbio. Rótulo: outro.
5. Livre: 3 minutos conversando e mexendo as mãos, sem gesto de comando. Rótulo: fluxo_sem_gesto.
Botões: pausar, repetir o último (marca invalido=1), marcar erro, encerrar e salvar.
Tempos: em relação ao início do vídeo, com requestVideoFrameCallback e performance.now como alternativa. Nunca use setTimeout como medida.
Arquivo anotacoes.csv com: id_participante,id_sessao,bloco,arquivo_video,indice_pedido,rotulo,mao,inicio_ms,fim_ms,inicio_frame,fim_frame,invalido,observacao
Também classes.json, metadados.json e LEIA-ME.txt.
Salvar: com showDirectoryPicker, crie sessao_<ID>_<data>_<luz>/ com vídeos e arquivos. Sem essa API, gere um ZIP com JSZip embutido. Grave em disco durante a captura sempre que possível.
Trate permissão negada, câmera desconectada, falta de espaço e aba em segundo plano. Use Wake Lock. Inclua um Modo de teste de 1 minuto.
Entregue o HTML completo em um único bloco de código e diga como testar.
```"""),
md("""## 7. Próximos passos

1. **Terminar o download do DriverMHG:** faltam 56 de 125 gravações, e o download retoma de onde parou.
2. **Preparar as gravações:** `src/preparar_drivermhg.py`, que também retoma.
3. **Experimento central:** treino do zero × pré-treino no Jester, com menos e mais pessoas no treino. Essa é a figura principal do paper.
4. **Falsos disparos no volante:** em fluxo contínuo, com as pessoas 19 a 25.
5. **Demais experimentos:** RGB × RGB + infravermelho, latência, vocabulário de 3, 5 e 7 gestos, e CNN 3D × CNN + GRU.
6. **Melhorias pendentes:** escolher o melhor modelo também pela falsa ativação, ajustar a taxa do transformer se ele for mantido, e testar uma janela mais curta para a latência.
7. **Dados no carro:** gravar com a página da seção 6.
8. **Com o professor:** confirmar o pré-treino próprio e revisar o termo de consentimento.
9. **Notebook:** atualizar o driver da NVIDIA e deixar a tampa aberta durante os treinos."""),
md("## 8. Diário\n\nRegistro feito ao longo do projeto, na ordem em que aconteceu. **[V]** é verificado, **[E]** é estimativa e **[?]** é ainda não confirmado."),
]
celulas += diario

notebook = nbf.v4.new_notebook(cells=celulas, metadata=antigo.metadata)
nbf.validate(notebook)
nbf.write(notebook, "notebook.ipynb")
print("células:", len(celulas), "| arquivos:", len(descricoes), "| diário:", len(diario))
