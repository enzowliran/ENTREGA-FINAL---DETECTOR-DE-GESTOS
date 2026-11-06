p='src/jester.py'; s=open(p,encoding='utf-8').read()
a='''classes_estaticas=(), aug_tempo=False):
        # Sorteia um trecho e uma velocidade, para a rede reconhecer gestos curtos na janela.
        self.aug_tempo = aug_tempo
'''
b='''classes_estaticas=(), aug_tempo=False, aug_espaco=False):
        # Sorteia um trecho e uma velocidade, para a rede reconhecer gestos curtos na janela.
        self.aug_tempo = aug_tempo
        # Reduz e desloca a cena, para a mão aparecer menor e fora do centro.
        self.aug_espaco = aug_espaco
'''
assert a in s; s=s.replace(a,b)
a='''            extensao = contagem * gerador.uniform(0.5, 1.0)
'''
b='''            extensao = contagem * gerador.uniform(0.4, 1.0)
'''
assert a in s; s=s.replace(a,b)
a='''        clipe = clipe[..., ::-1].transpose(0, 3, 1, 2).astype(np.float32) * brilho
'''
b='''        if self.treino and self.aug_espaco:
            escala = gerador.uniform(0.5, 1.0)
            lado_novo = max(int(self.tamanho * escala), 8)
            folga = self.tamanho - lado_novo
            y, x = int(gerador.integers(0, folga + 1)), int(gerador.integers(0, folga + 1))
            matriz = np.float32([[escala, 0, x], [0, escala, y]])
            clipe = np.stack([cv2.warpAffine(q, matriz, (self.tamanho, self.tamanho), flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_REPLICATE) for q in clipe])
        clipe = clipe[..., ::-1].transpose(0, 3, 1, 2).astype(np.float32) * brilho
'''
assert a in s; s=s.replace(a,b)
open(p,'w',encoding='utf-8').write(s)

p='src/treinar_jester.py'; s=open(p,encoding='utf-8').read()
a='''    analisador.add_argument("--suavizacao", '''
b='''    analisador.add_argument("--aug-espaco", action="store_true", help="Treina com a mão menor e fora do centro")
    analisador.add_argument("--suavizacao", '''
assert a in s; s=s.replace(a,b)
a='''                                aug_tempo=args.aug_tempo)
'''
b='''                                aug_tempo=args.aug_tempo, aug_espaco=args.aug_espaco)
'''
assert a in s; s=s.replace(a,b)
open(p,'w',encoding='utf-8').write(s)

import sys
sys.path.insert(0,'src')
import numpy as np, jester as J
_, tr, _, _ = J.ler_rotulos('data/Jester/rotulos/jester_labels')
c = J.ConjuntoJester('data/Jester', dict(list(tr.items())[:20]), treino=True, frac_sinteticos=0.0, classe_sem_gesto=0, aug_tempo=True, aug_espaco=True)
x, y = c.carregar(3, np.random.default_rng(1))
print('clipe', x.shape, x.dtype, round(float(x.min())), round(float(x.max())), 'classe', y)
