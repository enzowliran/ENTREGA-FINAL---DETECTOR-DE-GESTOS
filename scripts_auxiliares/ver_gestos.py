# Tiras para conferir visualmente o nome de cada gesto (mão direita, 3 exemplos de cada classe).
import sys, glob, os
sys.path.insert(0, 'src')
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import drivermhg as M

R = Path(sys.argv[1]); saida = Path(sys.argv[2]); saida.mkdir(parents=True, exist_ok=True)
a = M.ler_anotacao_gravacao(R)
print(a.groupby(['classe', 'gesto', 'mao']).size().to_string())
for classe in (1, 2, 3, 4, 5):
    s = a[a.classe == classe].head(2)
    fig, ax = plt.subplots(len(s), 8, figsize=(16, 2 * len(s)), squeeze=False)
    for i, l in enumerate(s.itertuples()):
        for j in range(8):
            f = l.inicio + (l.fim - l.inicio) * j // 7
            im = Image.open(R / 'rgb' / f'{f:05d}.jpg')
            ax[i][j].imshow(im); ax[i][j].axis('off'); ax[i][j].set_title(str(f), fontsize=7)
    fig.suptitle(f'classe {classe}'); fig.tight_layout(); fig.savefig(saida / f'c{classe}.png', dpi=80); plt.close(fig)
ok = sorted(os.listdir(R / 'rgb'))[:2]; print(ok, len(os.listdir(R / 'rgb')), 'ultimo fim', a.fim.max())
