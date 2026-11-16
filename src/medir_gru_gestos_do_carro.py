"""A GRU sobre os pontos e a combinação GRU + CNN (testadas, não usadas ao vivo) na régua dos gestos do carro.

Por quê: src/medir_gestos_do_carro.py mede o modelo final só nos 11 gestos que viram comando no carro. Para comparar
na mesma régua, este script mede ali também a GRU e as duas combinações com a CNN. Roda a GRU na CPU (rede pequena),
com os pesos e a normalização salvos em data/pontos/misto_modelos/gru_final.pt, e confere que o acerto nas 36 classes
bate com o já publicado (GRU 76,5%; números pela GRU 92,0%).

Uso:
    demo\python_pontos.bat src\medir_gru_gestos_do_carro.py
"""
import json
import pickle, sys
from pathlib import Path
import numpy as np, torch
RAIZ = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(Path(__file__).parent))
import comparar_misto as CM, treinar_pontos_jester as TP, medir_confianca_numeros as MC
PASTA = RAIZ / "data/pontos/misto_modelos"
jd = pickle.load(open(RAIZ / "data/pontos/jester_amostra.pkl", "rb"))
nomes = list(jd["nomes"]) + CM.EXTRAS; idx = {n: i for i, n in enumerate(nomes)}
am = [{"classe": c["classe"], "divisao": "treino" if c["divisao"] == "treino" else "teste", **{k: c[k] for k in ("img", "world", "present", "score", "altura_largura")}} for c in jd["clipes"]]
am += pickle.load(open(RAIZ / "data/pontos/misto.pkl", "rb"))
te = [s for s in am if s["divisao"] == "teste"]
y = np.load(PASTA / "teste_y.npy"); pc = np.load(PASTA / "cnn_pre_probs.npy")
assert all(idx[s["classe"]] == t for s, t in zip(te, y))
e = torch.load(PASTA / "gru_final.pt", map_location="cpu", weights_only=False)
assert list(e["nomes"]) == nomes
gru = TP.Rede(e["entrada"], len(nomes)); gru.load_state_dict(e["modelo"]); gru.eval()
X = [TP.atributos(s)[0] for s in te]
Xn = [np.where(x[:, -2:-1] > 0, (x - e["media"]) / e["desvio"], 0).astype(np.float32) for x in X]
pg = []
with torch.no_grad():
    for k in range(0, len(Xn), 256):
        xs, ms = zip(*[TP.fixar(x, None) for x in Xn[k:k + 256]])
        pg.append(gru(torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ms))).softmax(1).numpy())
pg = np.concatenate(pg)
num = [idx[n] for n in MC.NUMEROS]
def acc(p, sel=None):
    ok = p.argmax(1) == y
    return round(100 * float(ok.mean() if sel is None else ok[sel].mean()), 1)
media = 0.3 * pg + 0.7 * pc
pesos = media.copy(); pesos[:, num] = pg[:, num]
carro = np.isin(y, [idx[n] for n in ("One Finger", "Two Fingers", "Four Fingers", "Swiping Right", "Swiping Left", "Thumb Up", "Thumb Down", "Stop Sign", "Pushing Hand Away", "Drumming Fingers", "Pulling Hand In")])
nums = np.isin(y, [idx[n] for n in ("One Finger", "Two Fingers", "Four Fingers")])
saida = {"o_que_e": __doc__.split("\n\n")[0], "amostras_gestos_do_carro": int(carro.sum()), "modelos": {}}
for nome, p in (("GRU sobre os pontos", pg), ("GRU + CNN (média 0,3 / 0,7)", media), ("GRU + CNN, números pela GRU", pesos), ("CNN 3D pré-treinada", pc)):
    saida["modelos"][nome] = {"todas_as_36_classes_pct": acc(p), "gestos_do_carro_pct": acc(p, carro), "numeros_pct": acc(p, nums),
                              "movimentos_pct": acc(p, carro & ~nums)}
    print(nome, saida["modelos"][nome])
(RAIZ / "resultados/pontos/gestos_do_carro_gru.json").write_text(json.dumps(saida, ensure_ascii=False, indent=1), encoding="utf-8")
