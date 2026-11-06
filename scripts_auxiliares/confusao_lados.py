import sys
sys.path.insert(0, 'src')
from pathlib import Path
import numpy as np, torch
import jester as J, modelos as M, treinar_jester as T

estado = torch.load('resultados/jester/cnn_3d_medio/melhor.pt', map_location='cuda', weights_only=False)
nomes = estado['nomes']
m = M.criar_modelo(estado['arquitetura'], len(nomes)).cuda().eval(); m.load_state_dict(estado['modelo'])
_, _, val, _ = J.ler_rotulos('data/Jester/rotulos/jester_labels')
alvos = ['Swiping Left', 'Swiping Right', 'Stop Sign']
ids = [nomes.index(n) for n in alvos]
sub = {v: c for v, c in val.items() if c in ids}
c = J.ConjuntoJester('data/Jester', sub, treino=False)
pred, conf, verd = [], [], []
with torch.no_grad():
    for x, y in J.lotes(c, 48, seed=0, embaralhar=False):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p = m(T.preparar(x, 'cuda')).float().softmax(1)
        v, i = p.max(1); pred += i.tolist(); conf += v.tolist(); verd += y.tolist()
pred, conf, verd = map(np.array, (pred, conf, verd))
print('validacao completa, modelo medio')
for n, k in zip(alvos, ids):
    s = verd == k
    d = {nomes[j]: int(((pred == j) & s).sum()) for j in np.argsort(-np.bincount(pred[s], minlength=len(nomes)))[:4] if (pred[s] == j).any()}
    print(f'{n}: n={int(s.sum())} acerto={(pred[s] == k).mean():.3f} acerto_conf0.6={((pred[s] == k) & (conf[s] >= .6)).mean():.3f} conf_media_acertos={conf[s & (pred == k)].mean():.3f} -> {d}')
