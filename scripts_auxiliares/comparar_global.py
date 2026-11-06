import sys
import numpy as np
sys.path.insert(0, 'src')
import calibrar_por_gesto as P

d = dict(np.load('data/interno/janelas_validacao_cnn_3d.npz', allow_pickle=False))
n_cl = len(d['nomes'])
gestos = [i for i in range(n_cl) if i not in d['negativos'].tolist()]
minutos = float(d['minutos'])
n_vid = len(d['alvo'])
print('limiar n falsos f/min det err')
for n in (2, 3, 4):
    P.N_BASE = n
    for l in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8):
        lim = np.full(n_cl, np.inf)
        lim[gestos] = l
        f, _ = P.disparos(d['classe_neg'], d['conf_neg'], lim, n)
        a, e = P.avaliar_positivos(d, lim)
        print(f'{l:.2f} {n} {len(f):3d} {len(f)/minutos:.3f} {a.mean():.3f} {e.sum()/n_vid:.3f}')
