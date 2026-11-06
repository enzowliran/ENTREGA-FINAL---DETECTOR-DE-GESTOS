import nbformat as nbf
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
c = [
md("""# 01 · Exploração do DriverMHG

Contagens e durações usam as listas oficiais, com todas as pessoas. As tiras de frames usam uma única gravação (pessoa 1, gravação 1), a primeira que baixamos.

**Licença:** documentos publicados podem ter no máximo 10 frames e 2 clipes sem aprovação da TUM. Por isso as tiras são gravadas em `data/interno/`, fora do git, e não ficam embutidas neste notebook."""),
code("""import os, sys
from pathlib import Path
if Path.cwd().name == "notebooks":
    os.chdir("..")
sys.path.insert(0, "src")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
import drivermhg as M
pd.set_option("display.max_rows", 100, "display.width", 140)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
                     "axes.titlelocation": "left", "axes.titleweight": "bold", "figure.dpi": 110})
BASE = Path("data/DriverMHG")
PASTA_ANOTACAO = BASE / "anotacao/media/kop/760E6E913CA0F70B/PC_backup/datasets/DriverMHG/Annotation"
GRAVACAO = BASE / "amostra/media/kop/760E6E913CA0F70B/PC_backup/datasets/DriverMHG/Subject1/Rec1"
RESULTADOS = Path("resultados"); RESULTADOS.mkdir(exist_ok=True)
INTERNO = Path("data/interno"); INTERNO.mkdir(exist_ok=True)"""),
md("## Contagem por classe e divisão\n\nA divisão é a oficial: pessoas 1-18 no treino e 19-25 no teste."),
code("""anot = M.ler_listas(PASTA_ANOTACAO)
print("Segmentos:", len(anot), "| pessoas:", anot.pessoa.nunique())
print("Pessoas no treino:", sorted(anot[anot.subconjunto == "treino"].pessoa.unique()))
print("Pessoas no teste:", sorted(anot[anot.subconjunto == "teste"].pessoa.unique()))
assert not set(anot[anot.subconjunto == "treino"].pessoa) & set(anot[anot.subconjunto == "teste"].pessoa)
por_classe = anot.pivot_table(index=["gesto", "mao"], columns="subconjunto", values="inicio", aggfunc="count")
por_classe["total"] = por_classe.sum(axis=1)
por_classe.to_csv(RESULTADOS / "drivermhg_por_classe.csv")
por_classe"""),
md("Nos segmentos anotados, as classes são equilibradas. `None` e `Other` têm o mesmo peso dos gestos, então uma acurácia alta nesses recortes diz pouco sobre o uso contínuo."),
md("## Contagem por pessoa"),
code("""por_pessoa = anot.groupby(["pessoa", "subconjunto"]).agg(segmentos=("inicio", "count"), gravacoes=("gravacao", "nunique")).reset_index()
por_pessoa.to_csv(RESULTADOS / "drivermhg_por_pessoa.csv", index=False)
print(por_pessoa.segmentos.describe().round(1).to_string())
por_pessoa"""),
md("## Frames por gesto\n\nA 30 fps, cada frame equivale a 33 ms."),
code("""duracao = anot.groupby("gesto").n_frames.agg(["count", "mean", "median", "min", "max"]).round(1)
duracao["media_s"] = (duracao["mean"] / M.FPS).round(2)
duracao.to_csv(RESULTADOS / "drivermhg_duracao.csv")
print("Média geral:", round(anot.n_frames.mean(), 1), "frames =", round(anot.n_frames.mean() / M.FPS, 2), "s")
print(anot.groupby("subconjunto").n_frames.mean().round(1).to_string())
display(duracao)
fig, eixo = plt.subplots(figsize=(8, 3.6))
eixo.hist(anot.n_frames, bins=range(0, 91, 2), color="#2a78d6", rwidth=0.85)
eixo.set_xlabel("Duração do segmento (frames)"); eixo.set_ylabel("Segmentos")
eixo.set_title("A maioria dos segmentos dura cerca de 1 s")
fig.tight_layout(); fig.savefig("figuras/drivermhg_duracao.png", dpi=200); plt.show()"""),
md("Os segmentos do teste são um pouco mais longos que os do treino, o que pode indicar diferença entre as pessoas ou no jeito de anotar. É bom levar isso em conta ao escolher a janela temporal."),
md("""## A gravação como fluxo contínuo

No uso real não há recortes: o modelo roda o tempo todo. Aqui medimos quanto de uma gravação fica sem anotação. Esses intervalos são a base disponível para estimar falsos disparos."""),
code("""seq = M.ler_anotacao_gravacao(GRAVACAO)
n_frames = len(list((GRAVACAO / "rgb").glob("*.jpg")))
marcados = np.zeros(n_frames, dtype=bool)
for s in seq.itertuples():
    marcados[s.inicio:s.fim + 1] = True
gestos = seq[~seq.gesto.isin(["None", "Other"])]
print("Frames:", n_frames, "=", round(n_frames / M.FPS), "s")
print("Segmentos:", len(seq), "| gestos:", len(gestos), "| None + Other:", len(seq) - len(gestos))
print("Sem anotação:", int((~marcados).sum()), "frames =", round((~marcados).mean() * 100, 1), "% da gravação =", round((~marcados).sum() / M.FPS), "s")
print("Tempo entre gestos: mediana", round(gestos.inicio.diff().median() / M.FPS, 1), "s")
seq.head(10)"""),
md("Esses números valem para uma gravação só. A contagem do teste inteiro depende do download completo."),
md("""## Tiras de frames de cada gesto

Uma imagem por classe, com dois exemplos e 8 frames por exemplo, em RGB e infravermelho. As imagens ficam em `data/interno/`."""),
code("""def salvar_tira(classe, modalidade="rgb", exemplos=2, frames=8):
    amostras = seq[seq.classe == classe].head(exemplos)
    fig, eixos = plt.subplots(len(amostras), frames, figsize=(frames * 2, 1.6 * len(amostras)), squeeze=False)
    for i, s in enumerate(amostras.itertuples()):
        for j in range(frames):
            f = s.inicio + (s.fim - s.inicio) * j // (frames - 1)
            eixos[i][j].imshow(Image.open(GRAVACAO / modalidade / f"{f:05d}.jpg"))
            eixos[i][j].axis("off"); eixos[i][j].set_title(str(f), fontsize=7)
    gesto, mao = M.nome_classe(classe)
    fig.suptitle(f"{gesto}, mão {mao} ({modalidade})"); fig.tight_layout()
    caminho = INTERNO / f"{classe:02d}_{gesto}_{mao}_{modalidade}.png"
    fig.savefig(caminho, dpi=90); plt.close(fig)
    return caminho

caminhos = [salvar_tira(c, m) for c in range(1, 15) for m in ("rgb", "ir")]
print(len(caminhos), "tiras em", INTERNO)"""),
md("""## O que levar daqui

- **Divisão por pessoa:** as listas oficiais não têm nenhuma pessoa em comum entre treino e teste.
- **Gestos muito sutis:** a câmera filma de frente para o volante, as mãos são pequenas e o volante gira durante a direção. O movimento do gesto pode ser menor que o da própria direção.
- **Inversão horizontal proibida:** o README oficial avisa que ela troca a classe de alguns gestos.
- **Ainda sem confirmação:** a conferência visual da numeração das classes."""),
]
nb = nbf.v4.new_notebook(cells=c, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, "notebooks/01_exploracao.ipynb")
