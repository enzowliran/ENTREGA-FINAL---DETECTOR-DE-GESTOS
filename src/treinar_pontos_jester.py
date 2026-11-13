"""Teste pequeno: uma rede sobre os pontos da mão reconhece os 27 gestos do Jester?

Lê data/pontos/jester_amostra.pkl (src/extrair_pontos_jester.py). Treina só com os clipes em que o MediaPipe
achou a mão em pelo menos `--min-mao` dos quadros. Mede na amostra da validação oficial:
  - "com mão": só os clipes que passam no mesmo filtro (o teto do método);
  - "todos": todos os clipes da amostra (sem mão, a rede vê zeros), para comparar com a CNN 3D;
  - a CNN 3D (cnn_3d) nos mesmos clipes, com o mesmo protocolo da avaliação dela.

Rede: GRU bidirecional, 2 camadas de 128, com pooling por atenção. Atributos por quadro (mão principal):
forma relativa ao pulso em metros e na imagem, a mesma forma num referencial da palma, posição do pulso na
imagem (centrada na média do clipe), velocidade do pulso, tamanho aparente da mão e sua variação, presença.

Uso:
    demo\\python.bat src\\treinar_pontos_jester.py
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
RAIZ = Path(__file__).resolve().parent.parent
T = 40


def preencher_lacunas(x, presente, maximo=3):
    """Interpola no tempo falhas de até `maximo` quadros entre duas detecções."""
    x, presente = x.copy(), presente.copy()
    idx = np.nonzero(presente)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= maximo + 1:
            for k in range(a + 1, b):
                w = (k - a) / (b - a)
                x[k] = (1 - w) * x[a] + w * x[b]
                presente[k] = True
    return x, presente


def atributos(c):
    """[T, F] por clipe (mão do slot 0)."""
    a, l = c["altura_largura"] or (100, 176)
    pres = c["present"][:, 0]
    img = c["img"][:, 0].copy()
    img[..., 0] *= l
    img[..., 1] *= a
    img[..., 2] *= l
    junto = np.concatenate([img.reshape(len(img), -1), c["world"][:, 0].reshape(len(img), -1)], 1)
    junto, pres = preencher_lacunas(junto, pres)
    img, world = junto[:, :63].reshape(-1, 21, 3), junto[:, 63:].reshape(-1, 21, 3)
    n = len(img)
    tam = np.maximum(np.linalg.norm(img[:, 9, :2] - img[:, 0, :2], axis=-1), 1e-6)
    forma_img = (img[:, :, :2] - img[:, :1, :2]) / tam[:, None, None]
    w = world - world[:, :1]
    w = w / np.maximum(np.linalg.norm(w[:, 9], axis=-1), 1e-6)[:, None, None]
    # referencial da palma: e1 = pulso -> base do médio; e2 = normal da palma; e3 = e1 x e2
    e1 = w[:, 9] / np.maximum(np.linalg.norm(w[:, 9], axis=-1, keepdims=True), 1e-6)
    nrm = np.cross(w[:, 5], w[:, 17])
    nrm = nrm - (nrm * e1).sum(-1, keepdims=True) * e1
    e2 = nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
    e3 = np.cross(e1, e2)
    R = np.stack([e1, e2, e3], 1)  # [n, 3, 3]
    canon = np.einsum("nij,nkj->nki", R, w)
    orient = np.concatenate([e1, e2], 1)  # orientação da palma em relação à câmera (6)
    pulso = img[:, 0, :2] / a
    if pres.any():
        pulso = pulso - pulso[pres].mean(0)
    vel = np.zeros((n, 2), np.float32)
    vel[1:] = (img[1:, 0, :2] - img[:-1, 0, :2]) / tam[1:, None]
    dlog = np.zeros(n, np.float32)
    dlog[1:] = np.log(tam[1:] / tam[:-1])
    ambos = np.zeros(n, bool)
    ambos[1:] = pres[1:] & pres[:-1]
    vel[~ambos], dlog[~ambos] = 0, 0
    x = np.concatenate([forma_img.reshape(n, -1), w.reshape(n, -1), canon.reshape(n, -1), orient, pulso,
                        np.clip(vel, -3, 3), (tam / a)[:, None], np.clip(dlog, -1, 1)[:, None],
                        pres[:, None], c["score"][:, :1]], 1).astype(np.float32)
    x[~pres] = 0
    x[:, -2] = pres
    return np.nan_to_num(x), pres


def fixar(x, rng=None):
    """Recorta/completa para T quadros. No treino: velocidade 0,8-1,25x, recorte 80-100% e quadros zerados."""
    n = len(x)
    if rng is not None:
        ini_f, fim_f = 0.0, 1.0
        frac = rng.uniform(0.8, 1.0)
        ini_f = rng.uniform(0, 1 - frac)
        fim_f = ini_f + frac
        novo = int(np.clip(round(n * frac / rng.uniform(0.8, 1.25)), 8, T))
        idx = np.round(np.linspace(ini_f * (n - 1), fim_f * (n - 1), novo)).astype(int)
        x = x[idx].copy()
        x[rng.random(len(x)) < rng.uniform(0, 0.1)] = 0
    elif n > T:
        ini = (n - T) // 2
        x = x[ini:ini + T]
    saida = np.zeros((T, x.shape[1]), np.float32)
    mascara = np.zeros(T, bool)
    saida[:len(x)], mascara[:len(x)] = x, True
    return saida, mascara


class Rede(nn.Module):
    def __init__(self, entrada, classes):
        super().__init__()
        self.entrada = nn.Sequential(nn.Linear(entrada, 128), nn.ReLU())
        self.gru = nn.GRU(128, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.atencao = nn.Linear(256, 1)
        self.saida = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, classes))

    def forward(self, x, mascara):
        h = self.gru(self.entrada(x))[0]
        s = self.atencao(h).squeeze(-1).masked_fill(~mascara, -1e4)
        return self.saida((h * s.softmax(1)[..., None]).sum(1))


@torch.no_grad()
def cnn_nos_clipes(videos, classes):
    """CNN 3D (cnn_3d/melhor.pt) nos mesmos clipes, com o carregador e a avaliação de sempre."""
    import jester as J
    import modelos as M
    import treinar_jester as TJ
    e = torch.load(RAIZ / "resultados/jester/cnn_3d/melhor.pt", map_location="cuda", weights_only=False)
    m = M.criar_modelo(e["arquitetura"], len(e["nomes"])).cuda().eval()
    m.load_state_dict(e["modelo"])
    conj = J.ConjuntoJester(RAIZ / "data/Jester", dict(zip(videos, classes)), treino=False)
    ordem = {v: i for i, v in enumerate(videos)}
    pred = np.zeros(len(videos), int)
    k = 0
    for clipes, _ in J.lotes(conj, 64, seed=0, embaralhar=False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = m(TJ.preparar(clipes, "cuda")).float().argmax(1).cpu().numpy()
        for j, v in enumerate(conj.amostras[k:k + len(p)]):
            pred[ordem[v[0]]] = p[j]
        k += len(p)
    return pred, list(e["nomes"])


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--min-mao", type=float, default=0.7)
    a.add_argument("--epocas", type=int, default=80)
    a.add_argument("--saida", default="resultados/pontos/jester_coordenadas.json")
    args = a.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dados = pickle.load(open(RAIZ / "data/pontos/jester_amostra.pkl", "rb"))
    nomes = dados["nomes"]
    idx = {n: i for i, n in enumerate(nomes)}
    X, P, Y, D, V = [], [], [], [], []
    for c in dados["clipes"]:
        x, p = atributos(c)
        X.append(x), P.append(c["present"][:, 0].mean()), Y.append(idx[c["classe"]]), D.append(c["divisao"]), V.append(c["video"])
    P, Y, D, V = np.array(P), np.array(Y), np.array(D), np.array(V)
    ok = P >= args.min_mao
    tr = np.nonzero((D == "treino") & ok)[0]
    va_todos = np.nonzero(D == "validacao")[0]
    va_mao = np.nonzero((D == "validacao") & ok)[0]
    print(f"treino com mão: {len(tr)} de {(D == 'treino').sum()} clipes; validação com mão: {len(va_mao)} de {len(va_todos)}", flush=True)

    junto = np.concatenate([X[i][X[i][:, -2] > 0] for i in tr])
    media, desvio = junto.mean(0), junto.std(0) + 1e-6
    media[-2], desvio[-2] = 0, 1  # presença fica 0/1
    norm = lambda x: np.where(x[:, -2:-1] > 0, (x - media) / desvio, 0).astype(np.float32)
    Xn = [norm(x) for x in X]

    modelo = Rede(Xn[0].shape[1], len(nomes)).cuda()
    otim = torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-2)
    passos_epoca = int(np.ceil(len(tr) / 128))
    agenda = torch.optim.lr_scheduler.OneCycleLR(otim, 1e-3, total_steps=args.epocas * passos_epoca, pct_start=0.06)

    def lote(indices, treino):
        xs, ms = zip(*[fixar(Xn[i], rng if treino else None) for i in indices])
        x = torch.from_numpy(np.stack(xs)).cuda()
        if treino:
            x = x + 0.05 * torch.randn_like(x) * (x != 0)
        return x, torch.from_numpy(np.stack(ms)).cuda()

    @torch.no_grad()
    def prever(indices):
        modelo.eval()
        saida = []
        for k in range(0, len(indices), 512):
            x, m = lote(indices[k:k + 512], False)
            saida.append(modelo(x, m).float().cpu())
        modelo.train()
        return torch.cat(saida)

    t0, historico, melhor = time.time(), [], (-1, None)
    for ep in range(args.epocas):
        perm = rng.permutation(tr)
        for k in range(0, len(perm), 128):
            x, m = lote(perm[k:k + 128], True)
            y = torch.from_numpy(Y[perm[k:k + 128]]).cuda()
            perda = nn.functional.cross_entropy(modelo(x, m), y, label_smoothing=0.1)
            otim.zero_grad()
            perda.backward()
            otim.step()
            agenda.step()
        if True:
            acc = (prever(va_mao).argmax(1).numpy() == Y[va_mao]).mean() * 100
            historico.append({"epoca": ep + 1, "perda": round(perda.item(), 3), "val_com_mao": round(float(acc), 1)})
            print(f"  época {ep + 1}: perda {perda.item():.3f}, validação com mão {acc:.1f}% ({time.time() - t0:.0f} s)", flush=True)

    # Sem escolher época pela validação: usa o modelo do fim do treino (a amostra de validação é o teste aqui).
    p_todos = prever(va_todos)
    top1 = lambda p, i: round(float((p.argmax(1).numpy() == Y[i]).mean() * 100), 1)
    top5 = lambda p, i: round(float(p.topk(5, 1).indices.eq(torch.from_numpy(Y[i])[:, None]).any(1).float().mean() * 100), 1)
    p_mao = prever(va_mao)
    pred_cnn, nomes_cnn = cnn_nos_clipes(V[va_todos].tolist(), Y[va_todos].tolist())
    assert nomes_cnn == nomes, "ordem das classes da CNN diferente"
    cnn_ok = pred_cnn == Y[va_todos]
    em_mao = np.isin(va_todos, va_mao)
    por_classe = {}
    coord_ok = p_todos.argmax(1).numpy() == Y[va_todos]
    treino_por_classe = np.bincount(Y[tr], minlength=len(nomes))
    for n in nomes:
        s = Y[va_todos] == idx[n]
        por_classe[n] = {"clipes_teste": int(s.sum()), "clipes_teste_com_mao": int((s & em_mao).sum()),
                         "clipes_treino_com_mao": int(treino_por_classe[idx[n]]),
                         "acertos_coordenadas_todos": int(coord_ok[s].sum()), "acertos_cnn_3d_todos": int(cnn_ok[s].sum()),
                         "acertos_coordenadas_com_mao": int(coord_ok[s & em_mao].sum()), "acertos_cnn_3d_com_mao": int(cnn_ok[s & em_mao].sum())}
    print("\nclasse | teste | com mão | coord. (todos) | CNN 3D (todos) | coord. (com mão) | CNN 3D (com mão)")
    for n, c in por_classe.items():
        print(f"{n} | {c['clipes_teste']} | {c['clipes_teste_com_mao']} | {c['acertos_coordenadas_todos']} | {c['acertos_cnn_3d_todos']} | "
              f"{c['acertos_coordenadas_com_mao']} | {c['acertos_cnn_3d_com_mao']}")
    r = {"amostra": {"treino_com_mao": int(len(tr)), "treino_total": int((D == "treino").sum()),
                     "validacao_com_mao": int(len(va_mao)), "validacao_total": int(len(va_todos)), "min_mao": args.min_mao},
         "coordenadas": {"com_mao_top1": top1(p_mao, va_mao), "com_mao_top5": top5(p_mao, va_mao),
                         "todos_top1": top1(p_todos, va_todos), "todos_top5": top5(p_todos, va_todos)},
         "cnn_3d": {"todos_top1": round(float(cnn_ok.mean() * 100), 1), "com_mao_top1": round(float(cnn_ok[em_mao].mean() * 100), 1)},
         "parametros": sum(p.numel() for p in modelo.parameters()), "historico": historico, "por_classe": por_classe}
    (RAIZ / args.saida).write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in r.items() if k not in ("por_classe", "historico")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
