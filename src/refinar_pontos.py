"""Refina a classificação dos números pelos pontos da mão e compara regras, MLP, GRU e LSTM no mesmo critério.

Lê o cache de src/extrair_pontos.py (não roda o MediaPipe). Divisão, sem misturar pessoas:
  - treino das redes: pessoas de treino do IPN (menos 7), fotos de treino do HaGRID, Slovo sem sinal (treino);
  - escolha de limiares (regras e redes): 7 pessoas de treino do IPN separadas (as regras usam as 37);
  - medida final: as 13 pessoas de validação do IPN (as mesmas de sempre), Slovo sem sinal (teste),
    sinais "um" e "dois" do Slovo e do ASL Citizen, fotos de validação do HaGRID.

Critério de escolha: o maior acerto médio nos trechos de 1 e 2 dedos com no máximo `--alvo` falsos por minuto
nos trechos sem gesto.

Uso:
    demo\\python.bat src\\refinar_pontos.py
"""
import argparse
import itertools
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pontos_avaliacao as PA

RAIZ = PA.RAIZ
ARRAYS = PA.CACHE / "arrays.pkl"
SEPARADAS = 7  # pessoas de treino do IPN separadas para escolher limiares


# ---------- preparo ----------

def preparar():
    if ARRAYS.exists():
        return pickle.load(open(ARRAYS, "rb"))
    ipn, curtos, hagrid = PA.carregar()
    dados = {"ipn": {}, "curtos": [], "hagrid": None}
    for v, R in ipn.items():
        A = PA.arrays(R["quadros"])
        A.update(fps=R["fps"], trechos=R["trechos"], divisao=R["divisao"], pessoa=R["pessoa"], al=R["altura_largura"], duracao=R["duracao"])
        dados["ipn"][v] = A
    for it in curtos:
        A = PA.arrays(it["quadros"])
        A.update({k: it[k] for k in ("base", "classe", "divisao", "pessoa", "inicio", "fim", "fps", "duracao")}, al=it["altura_largura"])
        dados["curtos"].append(A)
    H = PA.arrays(hagrid)
    H.update(classe=np.array([h["classe"] for h in hagrid]), divisao=np.array([h["divisao"] for h in hagrid]), al=(1, 1), foto=True)
    dados["hagrid"] = H
    for A in [*dados["ipn"].values(), *dados["curtos"], H]:
        A["m"] = PA.medidas_vet(A["mundo"])
        A["direcao"] = PA.direcao_imagem(A["imagem"], A["al"])
        A["tamanho"] = PA.tamanho_mao(A["imagem"], A["al"])
    pickle.dump(dados, open(ARRAYS, "wb"))
    return dados


def grupos_ipn(dados):
    pessoas = sorted({A["pessoa"] for A in dados["ipn"].values() if A["divisao"] == "train"})
    separadas = set(np.random.default_rng(0).choice(pessoas, SEPARADAS, replace=False).tolist())
    g = {"treino": {}, "escolha": {}, "teste": {}}
    for v, A in dados["ipn"].items():
        chave = "teste" if A["divisao"] == "test" else ("escolha" if A["pessoa"] in separadas else "treino")
        g[chave][v] = A
    return g


def avaliar(conjunto_ipn, curtos, poses_fn, opcoes):
    """poses_fn(A) -> poses por quadro. Mede no IPN dado e nos vídeos curtos dados."""
    disp = {v: PA.simular(A, poses_fn(A), opcoes, A["al"]) for v, A in conjunto_ipn.items()}
    r = {"ipn": PA.medir_ipn(conjunto_ipn, disp)}
    if curtos:
        r["curtos"] = PA.medir_curtos(curtos, [PA.simular(A, poses_fn(A), opcoes, A["al"]) for A in curtos])
    return r


# ---------- regras: busca em grade ----------

_D = None


def _iniciar():
    global _D
    _D = grupos_ipn(preparar())


def _avaliar_regra(args):
    r, opcoes = args
    res = avaliar({**_D["treino"], **_D["escolha"]}, None, lambda A: PA.poses_regras(A, A["m"], A["direcao"], A["tamanho"], r), opcoes)
    return r, opcoes, res["ipn"]


def escolher(resultados, alvo):
    ok = [x for x in resultados if x[-1]["falsos_por_minuto"] <= alvo]
    if not ok:
        return min(resultados, key=lambda x: x[-1]["falsos_por_minuto"])
    return max(ok, key=lambda x: (x[-1]["media_certos"], -x[-1]["falsos_por_minuto"]))


def buscar_regras(alvo, processos):
    forma = {"reta": [0.80, 0.85, 0.90], "angulo": [50.0, 65.0], "dobrado": [None, 0.70, 0.60],
             "para_cima": [None, 60.0, 40.0], "conf": [0.0, 0.85], "polegar_reta": [0.95], "polegar_fora": [1.20], "tam_min": [0.0]}
    combos = [dict(zip(forma, v)) for v in itertools.product(*forma.values())]
    base = {"tempo_estavel": 0.5, "movimento": 0.5}
    t0 = time.time()
    with ProcessPoolExecutor(processos, initializer=_iniciar) as ex:
        etapa1 = list(ex.map(_avaliar_regra, [(r, base) for r in combos]))
        print(f"  regras, etapa 1: {len(etapa1)} combinações ({time.time() - t0:.0f} s)", flush=True)
        melhores = sorted(etapa1, key=lambda x: -x[-1]["media_certos"] + 100 * max(0, x[-1]["falsos_por_minuto"] - alvo))[:6]
        decisor = [{"tempo_estavel": t, "movimento": m} for t in (0.5, 0.8, 1.0) for m in (0.5, 0.3)]
        etapa2 = list(ex.map(_avaliar_regra, [(r, o) for r, _, _ in melhores for o in decisor]))
        print(f"  regras, etapa 2: {len(etapa2)} combinações ({time.time() - t0:.0f} s)", flush=True)
    todos = etapa1 + etapa2
    return escolher(todos, alvo), todos


# ---------- redes ----------

def atributos(A, velocidade):
    """Um vetor por quadro. Zeros quando não há mão (com o sinal 'tem mão' em 0)."""
    n = len(A["t"])
    mundo = A["mundo"] - A["mundo"][:, :1]
    escala = np.maximum(np.linalg.norm(mundo[:, 9], axis=-1, keepdims=True), 1e-6)
    mundo = mundo / escala[:, :, None]
    mundo[A["esquerda"], :, 0] *= -1  # mão esquerda espelhada
    m = A["m"]
    medidas = np.stack([m[f"{d}_reta"] for d in ("indicador", "médio", "anelar", "mínimo")]
                       + [m[f"{d}_angulo"] / 180 for d in ("indicador", "médio", "anelar", "mínimo")]
                       + [m["polegar_reta"], m["polegar_fora"] / 2], 1)
    a, l = A["al"] if A["al"] else (1, 1)
    xy = A["imagem"][..., :2] * np.array([l, a], np.float32)
    v = xy[:, 9] - xy[:, 0]
    tam = np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6)
    partes = [A["tem"][:, None].astype(np.float32), mundo.reshape(n, -1), medidas, A["direcao"] / 180, v / tam]
    if velocidade:
        vel = np.zeros((n, 3), np.float32)
        ambos = A["tem"][1:] & A["tem"][:-1]
        dpulso = (xy[1:, 0] - xy[:-1, 0]) / tam[1:]
        dtam = np.log(tam[1:, 0] / tam[:-1, 0])
        vel[1:, :2] = np.where(ambos[:, None], dpulso, 0)
        vel[1:, 2] = np.where(ambos, dtam, 0)
        partes.append(np.clip(vel, -3, 3))
    x = np.concatenate(partes, 1).astype(np.float32)
    x[~A["tem"]] = 0
    return np.nan_to_num(x)


def rotulos_ipn(A):
    y = np.full(len(A["t"]), -1, np.int64)
    f = A["t"] * A["fps"] + 1
    for ini, fim, r in A["trechos"]:
        if r in PA.ROTULO_IPN:
            y[(f >= ini) & (f <= fim)] = PA.ROTULO_IPN[r]
    return y


JANELA = 15  # 1 s a 15 quadros por segundo


def janelas(x, fim):
    """Janelas de JANELA quadros terminando em cada índice de `fim` (completa o começo com zeros)."""
    pad = np.concatenate([np.zeros((JANELA - 1, x.shape[1]), np.float32), x])
    return np.stack([pad[i:i + JANELA] for i in fim])


def treinar_rede(tipo, dados, g, passos, dispositivo, semente=0):
    import torch
    import torch.nn as nn

    seq = tipo != "mlp"
    rng = np.random.default_rng(semente)
    torch.manual_seed(semente)
    # IPN: quadros (ou janelas) rotulados das pessoas de treino
    ipn_x, ipn_y = [], []
    for A in g["treino"].values():
        x, y = atributos(A, seq), rotulos_ipn(A)
        idx = np.nonzero((y >= 0) & (A["tem"] | (y == 0)))[0][::2]
        ipn_x.append(janelas(x, idx) if seq else x[idx])
        ipn_y.append(y[idx])
    ipn_x, ipn_y = np.concatenate(ipn_x), np.concatenate(ipn_y)
    # Slovo sem sinal (treino): tudo "nenhum"
    sl_x = []
    for A in dados["curtos"]:
        if A["base"] == "Slovo sem sinal" and A["divisao"] == "train":
            x = atributos(A, seq)
            sl_x.append(janelas(x, np.arange(len(x))) if seq else x)
    sl_x = np.concatenate(sl_x)
    # HaGRID (treino): fotos com mão; nas redes de sequência, viram janelas paradas com tremor
    H = dados["hagrid"]
    sel = (H["divisao"] == "train") & H["tem"] & np.isin(H["classe"], list(PA.ROTULO_HAGRID))
    hx = atributos({k: (v[sel] if isinstance(v, np.ndarray) and len(v) == len(sel) else ({kk: vv[sel] for kk, vv in v.items()} if k == "m" else v)) for k, v in H.items()}, seq)
    hy = np.array([PA.ROTULO_HAGRID[c] for c in H["classe"][sel]])
    print(f"  {tipo}: IPN {len(ipn_y)} amostras, HaGRID {len(hy)}, Slovo {len(sl_x)}", flush=True)

    entrada = ipn_x.shape[-1]
    if tipo == "mlp":
        modelo = nn.Sequential(nn.Linear(entrada, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 5))
    else:
        class Rec(nn.Module):
            def __init__(self):
                super().__init__()
                cls = nn.LSTM if tipo == "lstm" else nn.GRU
                self.rec = cls(entrada, 96, num_layers=2, batch_first=True, dropout=0.2)
                self.saida = nn.Linear(96, 5)

            def forward(self, x):
                return self.saida(self.rec(x)[0][:, -1])
        modelo = Rec()
    modelo = modelo.to(dispositivo)
    otim = torch.optim.AdamW(modelo.parameters(), lr=2e-3, weight_decay=1e-4)
    agenda = torch.optim.lr_scheduler.OneCycleLR(otim, 2e-3, total_steps=passos)
    por_classe_ipn = {c: np.nonzero(ipn_y == c)[0] for c in (0, 1, 2)}
    por_classe_h = {c: np.nonzero(hy == c)[0] for c in range(5)}

    def lote_hagrid(n):
        c = rng.integers(0, 5, n)
        i = np.array([rng.choice(por_classe_h[k]) for k in c])
        x = hx[i]
        if seq:
            x = np.repeat(x[:, None], JANELA, 1) + rng.normal(0, 0.01, (n, JANELA, x.shape[-1])).astype(np.float32)
            x[..., 0] = 1
            x[..., -3:] = rng.normal(0, 0.02, (n, JANELA, 3))
            sem = rng.random((n, JANELA)) < 0.08
            x[sem] = 0
        else:
            x = x + rng.normal(0, 0.01, x.shape).astype(np.float32)
            x[:, 0] = 1
        return x, c

    modelo.train()
    for passo in range(passos):
        c = rng.integers(0, 3, 128)
        i = np.array([rng.choice(por_classe_ipn[k]) for k in c])
        xa, ya = ipn_x[i], ipn_y[i]
        xb, yb = lote_hagrid(96)
        j = rng.integers(0, len(sl_x), 32)
        x = np.concatenate([xa, xb, sl_x[j]])
        y = np.concatenate([ya, yb, np.zeros(32, np.int64)])
        x = torch.from_numpy(x).to(dispositivo)
        y = torch.from_numpy(y).to(dispositivo)
        perda = nn.functional.cross_entropy(modelo(x), y, label_smoothing=0.05)
        otim.zero_grad()
        perda.backward()
        otim.step()
        agenda.step()
        if (passo + 1) % 1000 == 0:
            print(f"    passo {passo + 1}/{passos}: perda {perda.item():.3f}", flush=True)
    modelo.eval()
    return modelo, seq


def probabilidades(modelo, seq, A, dispositivo):
    import torch
    x = atributos(A, seq)
    if seq and A.get("foto"):  # fotos do HaGRID: cada uma vira uma janela parada
        x = np.repeat(x[:, None], JANELA, 1)
    elif seq:
        x = janelas(x, np.arange(len(x)))
    saida = []
    with torch.no_grad():
        for k in range(0, len(x), 4096):
            saida.append(torch.softmax(modelo(torch.from_numpy(x[k:k + 4096]).to(dispositivo)), 1).cpu().numpy())
    return np.concatenate(saida) if saida else np.zeros((0, 5), np.float32)


def poses_de_prob(p, tem, limiar):
    k = p.argmax(1)
    return np.where(tem & (k > 0) & (p.max(1) >= limiar), k, -1)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--alvo", type=float, default=0.5, help="falsos por minuto máximos na escolha")
    a.add_argument("--processos", type=int, default=6)
    a.add_argument("--passos", type=int, default=6000)
    a.add_argument("--redes", nargs="*", default=["mlp", "gru", "lstm"])
    a.add_argument("--saida", default="resultados/pontos/refino.json")
    args = a.parse_args()
    t0 = time.time()
    dados = preparar()
    g = grupos_ipn(dados)
    print(f"IPN: {len(g['treino'])} vídeos de treino, {len(g['escolha'])} para escolha, {len(g['teste'])} de teste "
          f"({time.time() - t0:.0f} s)", flush=True)
    curtos_teste = [A for A in dados["curtos"] if A["divisao"] == "test"]
    H = dados["hagrid"]
    h_sel = (H["divisao"] == "test") & np.isin(H["classe"], list(PA.ROTULO_HAGRID))
    h_y = np.array([PA.ROTULO_HAGRID.get(c, -1) for c in H["classe"]])

    def acerto_hagrid(poses):
        p = np.where(poses > 0, poses, 0)
        ok = (p == h_y)[h_sel]
        return {"todas": round(100 * ok.mean(), 1),
                "numeros": round(100 * ok[h_y[h_sel] > 0].mean(), 1), "nao_numeros": round(100 * ok[h_y[h_sel] == 0].mean(), 1)}

    resultado = {"alvo_falsos_por_minuto": args.alvo, "metodos": {}}

    def registrar(nome, poses_fn, opcoes, extra):
        r = avaliar(g["teste"], curtos_teste, poses_fn, opcoes)
        r["hagrid_fotos"] = acerto_hagrid(poses_fn(H))
        r["opcoes_decisor"] = opcoes
        r.update(extra)
        resultado["metodos"][nome] = r
        print(f"\n== {nome}: IPN teste {r['ipn']['pct_trechos_certos']} (média {r['ipn']['media_certos']}), "
              f"{r['ipn']['falsos_por_minuto']} falsos/min {r['ipn']['falsos_por_classe']}; curtos {r['curtos']}; HaGRID {r['hagrid_fotos']}", flush=True)
        json.dump(resultado, open(RAIZ / args.saida, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)

    padrao = {"tempo_estavel": 0.5, "movimento": 0.5}
    regra = lambda r: (lambda A: PA.poses_regras(A, A["m"], A["direcao"], A["tamanho"], r))
    registrar("regras da Isabela, sem punho", regra(PA.REGRAS_ISABELA), padrao, {})

    melhor, todos = buscar_regras(args.alvo, args.processos)
    r, opcoes, na_escolha = melhor
    registrar("regras refinadas", regra(r), opcoes, {"parametros": r, "na_escolha_ipn_treino": na_escolha,
                                                     "combinacoes_testadas": len(todos)})
    resultado["grade_regras"] = [{"parametros": r, "decisor": o, **x} for r, o, x in todos]

    import torch
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    for tipo in args.redes:
        modelo, seq = treinar_rede(tipo, dados, g, args.passos, dispositivo)
        cache = {}

        def probs(A):
            chave = id(A)
            if chave not in cache:
                cache[chave] = probabilidades(modelo, seq, A, dispositivo)
            return cache[chave]

        candidatos = []
        for limiar in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            for t in (0.3, 0.5, 0.8):
                for mov in (0.5, 99.0):
                    o = {"tempo_estavel": t, "movimento": mov, "escala": 0.25 if mov < 10 else 99.0}
                    res = avaliar(g["escolha"], None, lambda A: poses_de_prob(probs(A), A["tem"], limiar), o)
                    candidatos.append((limiar, o, res["ipn"]))
        limiar, o, na_escolha = escolher(candidatos, args.alvo)
        registrar(tipo.upper(), lambda A: poses_de_prob(probs(A), A["tem"], limiar), o,
                  {"limiar": limiar, "na_escolha_ipn": na_escolha, "parametros": sum(p.numel() for p in modelo.parameters())})
        torch.save(modelo.state_dict() if seq else modelo, PA.CACHE / f"rede_{tipo}.pt")
    print(f"\nSalvo em {RAIZ / args.saida} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
