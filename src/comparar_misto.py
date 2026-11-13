"""Teste curto: CNN 3D x GRU sobre pontos, treinadas e testadas nas mesmas amostras de todas as bases juntas.

Bases: Jester (amostra de src/extrair_pontos_jester.py: 300 de treino e 100 de validação por classe), IPN Hand,
DriverMHG, HaGRID (números) e Slovo (sem sinal), montadas por src/montar_conjunto_misto.py. As duas redes recebem
as mesmas amostras de treino e são medidas nas mesmas de teste (pessoas fora do treino em todas as bases).

Modelos:
  - GRU sobre os pontos (do zero), como em src/treinar_pontos_jester.py;
  - CNN 3D (cnn_3d) do zero, nas mesmas amostras;
  - CNN 3D partindo do modelo treinado no Jester inteiro (cnn_3d/melhor.pt), com uma cabeça nova: é como usamos a
    CNN no projeto, mas ela viu muito mais Jester que a GRU.

Uso:
    demo\\python.bat src\\comparar_misto.py
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import jester as J
import modelos as M
import treinar_jester as TJ
import treinar_pontos_jester as TP

RAIZ = Path(__file__).resolve().parent.parent
PASTA = RAIZ / "data/pontos/misto_modelos"  # pontos de retomada e probabilidades no teste (fora do git)
EXTRAS = ["One Finger", "Two Fingers", "Three Fingers", "Four Fingers",
          "Volante: FlickDown", "Volante: FlickUp", "Volante: SwipeLeft", "Volante: SwipeRight", "Volante: Tap"]


class ConjuntoComb:
    """Amostras do Jester (lidas do arquivo do Jester) e das outras bases (JPGs prontos), para a CNN."""

    def __init__(self, amostras, treino, n_frames=16, tamanho=96):
        self.amostras, self.treino, self.n, self.tam = amostras, treino, n_frames, tamanho
        jester = {s["video"]: s["y"] for s in amostras if s["fonte"] == "Jester"}
        self.cj = J.ConjuntoJester(RAIZ / "data/Jester", jester, treino=treino, aug_tempo=treino) if jester else None
        self.pos = {a[0]: i for i, a in enumerate(self.cj.amostras)} if self.cj else {}

    def __len__(self):
        return len(self.amostras)

    def carregar(self, posicao, g):
        s = self.amostras[posicao]
        if s["fonte"] == "Jester":
            return self.cj.carregar(self.pos[s["video"]], g)[0], s["y"]
        jpgs = s["jpgs"]
        pontos = np.linspace(0, len(jpgs), self.n + 1)
        if self.treino:
            idx = (pontos[:-1] + g.random(self.n) * (pontos[1:] - pontos[:-1])).astype(int)
        else:
            idx = ((pontos[:-1] + pontos[1:]) / 2).astype(int)
        imgs = [cv2.imdecode(np.frombuffer(jpgs[min(i, len(jpgs) - 1)], np.uint8), cv2.IMREAD_COLOR) for i in idx]
        lado = imgs[0].shape[0]
        if self.treino:
            r = int(lado * g.uniform(0.8, 1.0))
            y0, x0, br = int(g.integers(0, lado - r + 1)), int(g.integers(0, lado - r + 1)), g.uniform(0.8, 1.2)
        else:
            r, y0, x0, br = lado, 0, 0, 1.0
        clipe = np.stack([cv2.resize(im[y0:y0 + r, x0:x0 + r], (self.tam, self.tam), interpolation=cv2.INTER_AREA) for im in imgs])
        return clipe[..., ::-1].transpose(0, 3, 1, 2).astype(np.float32) * br, s["y"]


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--epocas-gru", type=int, default=80)
    a.add_argument("--epocas-cnn", type=int, default=15)
    a.add_argument("--epocas-cnn-pre", type=int, default=6)
    a.add_argument("--saida", default="resultados/pontos/misto_comparacao.json")
    a.add_argument("--modelos", nargs="*", default=["gru", "cnn", "cnn_pre"])
    args = a.parse_args()
    PASTA.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    torch.backends.cudnn.benchmark = True

    jd = pickle.load(open(RAIZ / "data/pontos/jester_amostra.pkl", "rb"))
    nomes = list(jd["nomes"]) + EXTRAS
    idx = {n: i for i, n in enumerate(nomes)}
    amostras = [{"fonte": "Jester", "video": c["video"], "y": idx[c["classe"]], "divisao": "treino" if c["divisao"] == "treino" else "teste",
                 **{k: c[k] for k in ("img", "world", "present", "score", "altura_largura")}} for c in jd["clipes"]]
    for s in pickle.load(open(RAIZ / "data/pontos/misto.pkl", "rb")):
        s["y"] = idx[s["classe"]]
        amostras.append(s)
    for s in amostras:
        s["mao"] = float(s["present"][:, 0].mean())
    tr = [i for i, s in enumerate(amostras) if s["divisao"] == "treino"]
    te = [i for i, s in enumerate(amostras) if s["divisao"] == "teste"]
    Y = np.array([s["y"] for s in amostras])
    fontes = sorted({s["fonte"] for s in amostras})
    print(f"{len(nomes)} classes; treino {len(tr)}, teste {len(te)} amostras; fontes: "
          + ", ".join(f"{f} {sum(amostras[i]['fonte'] == f for i in tr)}/{sum(amostras[i]['fonte'] == f for i in te)}" for f in fontes), flush=True)
    resultado = {"classes": nomes, "modelos": {}}

    # ---------- GRU sobre os pontos ----------
    X = [TP.atributos(s)[0] for s in amostras]
    junto = np.concatenate([X[i][X[i][:, -2] > 0] for i in tr])
    media, desvio = junto.mean(0), junto.std(0) + 1e-6
    media[-2], desvio[-2] = 0, 1
    Xn = [np.where(x[:, -2:-1] > 0, (x - media) / desvio, 0).astype(np.float32) for x in X]
    gru = TP.Rede(Xn[0].shape[1], len(nomes)).cuda()
    otim = torch.optim.AdamW(gru.parameters(), lr=1e-3, weight_decay=1e-2)
    passos = int(np.ceil(len(tr) / 128))
    agenda = torch.optim.lr_scheduler.OneCycleLR(otim, 1e-3, total_steps=args.epocas_gru * passos, pct_start=0.06)

    def lote_gru(ind, treino):
        xs, ms = zip(*[TP.fixar(Xn[i], rng if treino else None) for i in ind])
        x = torch.from_numpy(np.stack(xs)).cuda()
        if treino:
            x = x + 0.05 * torch.randn_like(x) * (x != 0)
        return x, torch.from_numpy(np.stack(ms)).cuda()

    @torch.no_grad()
    def prever_gru(ind):
        gru.eval()
        out = torch.cat([gru(*lote_gru(ind[k:k + 512], False)).float().cpu() for k in range(0, len(ind), 512)])
        gru.train()
        return out

    t0 = time.time()
    probs = {}
    feito = PASTA / "gru_probs.npy"
    for ep in (range(args.epocas_gru) if "gru" in args.modelos and not feito.exists() else []):
        perm = rng.permutation(tr)
        for k in range(0, len(perm), 128):
            x, m = lote_gru(perm[k:k + 128], True)
            perda = nn.functional.cross_entropy(gru(x, m), torch.from_numpy(Y[perm[k:k + 128]]).cuda(), label_smoothing=0.1)
            otim.zero_grad()
            perda.backward()
            otim.step()
            agenda.step()
        acc = (prever_gru(te).argmax(1).numpy() == Y[te]).mean() * 100
        print(f"GRU época {ep + 1}/{args.epocas_gru}: teste {acc:.1f}% ({time.time() - t0:.0f} s)", flush=True)
    if "gru" in args.modelos:
        if not feito.exists():
            np.save(feito, prever_gru(te).softmax(1).numpy())
        probs["GRU sobre pontos"] = np.load(feito)
        print("GRU pronta (probabilidades salvas)" if feito.exists() else "", flush=True)

    # ---------- CNN 3D ----------
    c_tr, c_te = ConjuntoComb([amostras[i] for i in tr], True), ConjuntoComb([amostras[i] for i in te], False)

    @torch.no_grad()
    def prever_cnn(m):
        m.eval()
        p = []
        for clipes, _ in J.lotes(c_te, 64, seed=0, embaralhar=False):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p.append(m(TJ.preparar(clipes, "cuda")).float().softmax(1).cpu())
        m.train()
        return torch.cat(p).numpy()

    def treinar_cnn(m, epocas, lr, rotulo, chave):
        """Salva um ponto de retomada a cada época; se o treino já terminou, só lê as probabilidades salvas."""
        final = PASTA / f"{chave}_probs.npy"
        if final.exists():
            print(f"{rotulo}: já treinada, lendo as probabilidades salvas", flush=True)
            return np.load(final)
        otim = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
        passos = len(c_tr) // 48 + 1
        agenda = torch.optim.lr_scheduler.OneCycleLR(otim, lr, total_steps=epocas * passos, pct_start=0.1)
        retomada, primeira = PASTA / f"{chave}_ultimo.pt", 0
        if retomada.exists():
            e = torch.load(retomada, map_location="cuda", weights_only=False)
            m.load_state_dict(e["modelo"]); otim.load_state_dict(e["otimizador"]); agenda.load_state_dict(e["agendador"])
            primeira = e["epoca"] + 1
            print(f"{rotulo}: retomando da época {primeira + 1}", flush=True)
        t0 = time.time()
        for ep in range(primeira, epocas):
            certos = total = 0
            for clipes, y in J.lotes(c_tr, 48, seed=ep):
                y = y.cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    s = m(TJ.preparar(clipes, "cuda"))
                    perda = nn.functional.cross_entropy(s.float(), y, label_smoothing=0.1)
                otim.zero_grad(set_to_none=True)
                perda.backward()
                otim.step()
                if agenda.last_epoch < epocas * passos - 1:
                    agenda.step()
                certos += (s.argmax(1) == y).sum().item()
                total += y.numel()
            p = prever_cnn(m)
            acc = (p.argmax(1) == Y[te]).mean() * 100
            print(f"{rotulo} época {ep + 1}/{epocas}: treino {100 * certos / total:.1f}%, teste {acc:.1f}% ({time.time() - t0:.0f} s)", flush=True)
            torch.save({"modelo": m.state_dict(), "otimizador": otim.state_dict(), "agendador": agenda.state_dict(), "epoca": ep}, retomada)
        np.save(final, p)
        torch.save({"modelo": m.state_dict(), "nomes": nomes, "arquitetura": "cnn_3d"}, PASTA / f"{chave}_final.pt")
        return p

    if "cnn" in args.modelos:
        cnn = M.criar_modelo("cnn_3d", len(nomes)).cuda()
        probs["CNN 3D do zero"] = treinar_cnn(cnn, args.epocas_cnn, 1e-3, "CNN do zero", "cnn")
        del cnn
    e = torch.load(RAIZ / "resultados/jester/cnn_3d/melhor.pt", map_location="cuda", weights_only=False)
    pre = M.criar_modelo("cnn_3d", len(e["nomes"])).cuda()
    pre.load_state_dict(e["modelo"])
    velha = pre.cabeca
    nova = nn.Linear(velha.in_features, len(nomes)).cuda() if isinstance(velha, nn.Linear) else None
    if nova is None:  # cabeça com dropout + linear
        lin = [mod for mod in velha.modules() if isinstance(mod, nn.Linear)][-1]
        nova = nn.Linear(lin.in_features, len(nomes)).cuda()
        with torch.no_grad():
            nova.weight[:27], nova.bias[:27] = lin.weight, lin.bias
        partes = [mod for mod in velha.children()]
        pre.cabeca = nn.Sequential(*partes[:-1], nova)
    else:
        with torch.no_grad():
            nova.weight[:27], nova.bias[:27] = velha.weight, velha.bias
        pre.cabeca = nova
    if "cnn_pre" in args.modelos:
        probs["CNN 3D partindo do Jester inteiro"] = treinar_cnn(pre, args.epocas_cnn_pre, 3e-4, "CNN pré-treinada", "cnn_pre")
    pred = {k: v.argmax(1) for k, v in probs.items()}
    np.save(PASTA / "teste_y.npy", Y[te])
    np.save(PASTA / "teste_mao.npy", np.array([amostras[i]["mao"] for i in te]))
    np.save(PASTA / "teste_fonte.npy", np.array([amostras[i]["fonte"] for i in te]))

    # ---------- tabelas ----------
    Yt = Y[te]
    mao = np.array([amostras[i]["mao"] >= 0.7 for i in te])
    fonte = np.array([amostras[i]["fonte"] for i in te])
    for nome, p in pred.items():
        ok = p == Yt
        resultado["modelos"][nome] = {
            "todos": round(100 * ok.mean(), 1), "com_mao": round(100 * ok[mao].mean(), 1),
            "por_fonte": {f: {"n": int((fonte == f).sum()), "todos": round(100 * ok[fonte == f].mean(), 1),
                              "com_mao": round(100 * ok[(fonte == f) & mao].mean(), 1) if ((fonte == f) & mao).any() else None}
                          for f in fontes}}
    resultado["por_classe"] = {}
    for n in nomes:
        s = Yt == idx[n]
        if s.any():
            resultado["por_classe"][n] = {"teste": int(s.sum()), "com_mao": int((s & mao).sum()),
                                          **{k: int((p[s] == Yt[s]).sum()) for k, p in pred.items()}}
    resultado["amostras_com_mao_por_fonte"] = {f: round(100 * mao[fonte == f].mean(), 1) for f in fontes}
    (RAIZ / args.saida).write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nRESUMO", json.dumps({k: v for k, v in resultado.items() if k in ("modelos", "amostras_com_mao_por_fonte")}, ensure_ascii=False))
    print("\nclasse | teste | com mão | " + " | ".join(pred))
    for n, c in resultado["por_classe"].items():
        print(f"{n} | {c['teste']} | {c['com_mao']} | " + " | ".join(str(c[k]) for k in pred))


if __name__ == "__main__":
    main()
