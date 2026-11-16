"""Acerto da configuração usada ao vivo (CNN 3D pré-treinada + contagem de dedos) no teste misto.

Mesmo teste das outras linhas da tabela (3.642 amostras de 5 bases, pessoas fora do treino), sem rodar nenhuma rede:
  - CNN: as notas já salvas por src/comparar_misto.py (data/pontos/misto_modelos/cnn_pre_probs.npy, a mesma CNN do
    modelo final), na ordem das amostras de teste;
  - contagem de dedos: a mesma simulação de src/medir_confianca_numeros.py (mão parada, mesma contagem de 1 a 4 dedos em
    5 de 6 quadros, dispara direto), sobre os pontos já extraídos de cada amostra.
Decisão, como no player: se a contagem disparou um número na amostra, a resposta é esse número; senão, a classe de
maior nota da CNN sem as classes de número.

Uso:
    demo\\python.bat src\\medir_modelo_ao_vivo.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import medir_confianca_numeros as MC

RAIZ = Path(__file__).resolve().parent.parent
PASTA = RAIZ / "data/pontos/misto_modelos"


def main():
    # mesma ordem de src/comparar_misto.py: clipes do Jester, depois o conjunto misto; teste = divisão "teste"
    jd = pickle.load(open(RAIZ / "data/pontos/jester_amostra.pkl", "rb"))
    nomes = list(jd["nomes"]) + ["One Finger", "Two Fingers", "Three Fingers", "Four Fingers",
                                 "Volante: FlickDown", "Volante: FlickUp", "Volante: SwipeLeft", "Volante: SwipeRight", "Volante: Tap"]
    amostras = [{"fonte": "Jester", "classe": c["classe"], "divisao": "treino" if c["divisao"] == "treino" else "teste",
                 **{k: c[k] for k in ("img", "world", "present", "altura_largura")}} for c in jd["clipes"]]
    amostras += pickle.load(open(RAIZ / "data/pontos/misto.pkl", "rb"))
    te = [s for s in amostras if s["divisao"] == "teste"]
    y = np.load(PASTA / "teste_y.npy")
    fonte = np.load(PASTA / "teste_fonte.npy")
    probs = np.load(PASTA / "cnn_pre_probs.npy")
    assert len(te) == len(y) == len(probs), (len(te), len(y), len(probs))
    import comparar_misto as CM  # para conferir que a lista de classes é a mesma usada no treino
    nomes = list(jd["nomes"]) + CM.EXTRAS
    idx = {n: i for i, n in enumerate(nomes)}
    assert all(idx[s["classe"]] == t for s, t in zip(te, y)), "ordem das amostras diferente da do treino"
    assert all(s["fonte"] == f for s, f in zip(te, fonte))

    numeros = [idx[n] for n in MC.NUMEROS]
    sem_num = probs.copy()
    sem_num[:, numeros] = 0
    pred_cnn = sem_num.argmax(1)
    regra = [MC.disparo(s) for s in te]
    pred = np.array([idx[r] if r else c for r, c in zip(regra, pred_cnn)])
    certo = pred == y
    e_num = np.isin(y, numeros)
    so_cnn = probs.argmax(1) == y

    def pct(m):
        return round(100 * float(m.mean()), 1) if m.size else None

    r = {
        "o_que_e": "Acerto da configuração usada ao vivo (CNN 3D pré-treinada + contagem de dedos, número direto em 5 de 6 "
                   "quadros com a mão parada; senão a CNN sem as classes de número) no teste misto. Sem rodar rede: notas "
                   "salvas da CNN e pontos já extraídos. A metade das pessoas do HaGRID usada para ajustar os limiares da "
                   "contagem está incluída (como nas outras linhas); o acerto sem ela também é dado.",
        "amostras": int(len(y)),
        "acerto_pct": pct(certo),
        "acerto_numeros_pct": pct(certo[e_num]),
        "acerto_sem_numeros_pct": pct(certo[~e_num]),
        "comparacao_cnn_pre_sozinha_pct": pct(so_cnn),
        "por_fonte_pct": {f: pct(certo[fonte == f]) for f in sorted(set(fonte))},
        "regra_disparou_em_pct_das_amostras": round(100 * sum(r is not None for r in regra) / len(regra), 1),
    }
    hag = np.array([s["fonte"] == "HaGRID" and MC.metade(s.get("pessoa")) == 0 for s in te])
    r["acerto_sem_hagrid_de_ajuste_pct"] = pct(certo[~hag])
    saida = RAIZ / "resultados/pontos/modelo_ao_vivo_misto.json"
    saida.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
