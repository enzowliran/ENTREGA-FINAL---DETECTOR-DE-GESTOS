"""Falsos disparos por minuto da configuração usada ao vivo (CNN 3D pré-treinada + contagem de dedos), em vídeo contínuo.

Mesmo critério de src/fluxo_combinado.py (as outras linhas da tabela): vídeos do IPN das 13 pessoas de validação,
quadros a 12 por segundo, janelas de 1, 1,5 e 3 s, uma análise a cada 0,15 s; disparo dentro de um trecho "sem gesto"
(D0X) = falso. Também mede os vídeos "sem sinal" do Slovo (teste).

Em cada análise, como em demo/player.py (laco_pontos):
  - contagem de dedos nos pontos dos últimos 6 quadros (src/medir_confianca_numeros.py: mão parada e o mesmo número
    de 1 a 4 em 5 de 6 quadros). Se der número, ele vai direto para o gatilho (sem nota; uma vez enquanto mantido;
    pausa de 1,2 s) e a média da CNN é zerada;
  - senão, a CNN sem as classes de número (renormalizada), a janela com o gesto mais confiante, média das 2 últimas
    análises e o gatilho >= 0,7 em 3 análises seguidas.
Variantes: a contagem confirmada em 1 (ao vivo hoje), 3 ou 5 análises seguidas; e a CNN sozinha, para conferir que
este script reproduz o valor já medido (0,19 / min).
Dois conjuntos de gestos que podem disparar: todos os gestos (como nas outras linhas da tabela) e só os comandos do
carro (demo/player.py, COMANDOS_CARRO3D).

Roda a CNN na GPU (inferência, sem treino).

Uso:
    demo\\python_pontos.bat src\\fluxo_ao_vivo.py
"""
import collections
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import fluxo_combinado as FC
import fluxo_ipn as F
import medir_confianca_numeros as MC
import modelos as M
import treinar_jester as T

RAIZ = Path(__file__).resolve().parent.parent
CARRO = {"Swiping Left", "Swiping Right", "Thumb Up", "Thumb Down", "Drumming Fingers", "Stop Sign", "Pushing Hand Away",
         "Pulling Hand In", "One Finger", "Two Fingers", "Four Fingers"}
PAUSA, SOLTAR = 1.2, 0.5


class Gatilho:
    """O gatilho de demo/player.py (com o caminho da regra), sem repetição de polegar."""

    def __init__(self, comandos):
        self.comandos = comandos
        self.atual, self.contagem, self.fim_pausa, self.armado = None, 0, 0.0, True
        self.regra_nome, self.regra_visto, self.regra_disparou = None, -1e9, None

    def cnn(self, nome, conf, agora):
        if nome not in self.comandos or conf < 0.7:
            self.atual, self.contagem, self.armado = None, 0, True
            return None
        if nome != self.atual:
            self.atual, self.contagem, self.armado = nome, 0, True
        self.contagem += 1
        if self.armado and self.contagem >= 3 and agora >= self.fim_pausa:
            self.armado, self.fim_pausa = False, agora + PAUSA
            return nome
        return None

    def regra(self, nome, agora):
        mantido = nome == self.regra_nome and agora - self.regra_visto < SOLTAR
        self.regra_nome, self.regra_visto = nome, agora
        if not mantido:
            self.regra_disparou = None
        self.atual, self.contagem = nome, 1
        if nome not in self.comandos or self.regra_disparou == nome or agora < self.fim_pausa:
            self.armado = self.regra_disparou != nome
            return None
        self.regra_disparou, self.armado, self.fim_pausa = nome, False, agora + PAUSA
        return nome


def numeros_da_regra(pontos, tempos):
    """Número dado pela contagem de dedos em cada análise (ou None)."""
    img, world, pres, al = pontos["img"][:, 0], pontos["world"][:, 0], pontos["present"][:, 0], pontos["altura_largura"]
    saida = []
    for t in tempos:
        f = min(int(round(t * FC.QPS)), len(pres)) - 1
        if f < MC.N - 1:
            saida.append(None)
            continue
        n = MC.contagem(world, pres, f) if MC.parada(img, pres, al, f) else None
        saida.append(n if n in MC.NUMEROS else None)
    return saida


# Veto da CNN (src/medir_veto_outros.py): gestos que não são comando no carro somados em "Outros gestos"; se a CNN
# apontar "Outros gestos" ou um movimento do carro, o número da contagem é ignorado.
MOVIMENTOS = {"Swiping Left", "Swiping Right", "Thumb Up", "Thumb Down", "Drumming Fingers", "Stop Sign", "Pushing Hand Away", "Pulling Hand In"}
FICAM = MOVIMENTOS | set(MC.NUMEROS) | {"No gesture"}


def vetada(q, nomes, sem_gesto=False):
    """q: notas da CNN (36 classes) numa janela. True se a CNN aponta 'Outros gestos' ou um movimento
    (e, com sem_gesto, também 'No gesture')."""
    outros = sum(q[i] for i, n in enumerate(nomes) if n not in FICAM)
    resto = [(q[i], n) for i, n in enumerate(nomes) if n in FICAM]
    melhor, nome = max(resto)
    return outros > melhor or nome in MOVIMENTOS or (sem_gesto and nome == "No gesture")


def simular(tempos, por, pc, regra, nomes, gestos, numeros, comandos, confirmar, usar_regra=True, veto=None, sem_gesto=False):
    g, hist, saida = Gatilho(comandos), collections.deque(maxlen=2), []
    if veto:   # janela usada no veto: a mais curta (1 s) ou a mais longa disponível
        blocos, k = [], 0
        for n in por:
            blocos.append(pc[k] if veto == "curta" else pc[k + n - 1])
            k += n
        regra = [None if (r is not None and vetada(b, nomes, sem_gesto)) else r for r, b in zip(regra, blocos)]
    p = pc.copy()
    if usar_regra:
        p[:, numeros] = 0
        p /= np.maximum(p.sum(1, keepdims=True), 1e-9)
    escolhidas = FC.escolher(por, p, gestos)
    seguidas, ultimo = 0, None
    for t, q, n in zip(tempos, escolhidas, regra if usar_regra else [None] * len(tempos)):
        seguidas = seguidas + 1 if (n is not None and n == ultimo) else (1 if n is not None else 0)
        ultimo = n
        if n is not None and seguidas >= confirmar:
            hist.clear()
            d = g.regra(n, t)
        else:
            hist.append(q)
            m = np.mean(hist, axis=0)
            k = int(m.argmax())
            d = g.cnn(nomes[k], float(m[k]), t)
        if d:
            saida.append((t, d))
    return saida


def main():
    t0 = time.time()
    e = torch.load(FC.PASTA / "cnn_pre_final.pt", map_location="cuda", weights_only=False)
    nomes = list(e["nomes"])
    cnn = M.criar_modelo("cnn_3d", len(nomes)).cuda().eval()
    cnn.load_state_dict(e["modelo"])
    gestos = [i for i, n in enumerate(nomes) if n not in T.SEM_GESTO]
    numeros = [nomes.index(n) for n in MC.NUMEROS]
    todos = {n for n in nomes if n not in T.SEM_GESTO}
    variantes = {"CNN pré-treinada sozinha": dict(usar_regra=False, confirmar=1)}
    variantes["CNN 3D + contagem de dedos (confirmada em 1 análise)"] = dict(usar_regra=True, confirmar=1)
    variantes["CNN 3D + contagem de dedos + veto da CNN (janela mais longa)"] = dict(usar_regra=True, confirmar=1, veto="longa")
    variantes["CNN 3D + contagem de dedos + veto da CNN com Sem gesto (janela mais longa)"] = dict(usar_regra=True, confirmar=1, veto="longa", sem_gesto=True)
    variantes["CNN 3D + contagem de dedos + veto da CNN com Sem gesto (janela de 1 s)"] = dict(usar_regra=True, confirmar=1, veto="curta", sem_gesto=True)
    conjuntos = {"todos os gestos": todos, "só comandos do carro": CARRO}

    ipn = pickle.load(open(RAIZ / "data/pontos/ipn.pkl", "rb"))
    videos = sorted(v for v, r in ipn.items() if r["divisao"] == "test")
    R = {(v, c): {"falsos": collections.Counter(), "certos": {"B0A": 0, "B0B": 0}} for v in variantes for c in conjuntos}
    sem_gesto, n_apontar = 0.0, {"B0A": 0, "B0B": 0}
    alvo = {"B0A": "One Finger", "B0B": "Two Fingers"}
    for k, v in enumerate(videos, 1):
        reg = ipn[v]
        quadros, fps = F.quadros_12(RAIZ / f"data/IPN/videos/videos/{v}.avi")
        pts = FC.pontos_12(reg)
        tempos, por, pc, _ = FC.analisar(cnn, None, None, quadros, None, gestos)
        regra = numeros_da_regra(pts, tempos)
        trechos = reg["trechos"]
        sem_gesto += sum(f - i + 1 for i, f, r in trechos if r == "D0X") / fps / 60
        for r in alvo:
            n_apontar[r] += sum(1 for *_, x in trechos if x == r)
        for nv, cfg in variantes.items():
            for nc, cmds in conjuntos.items():
                disp = simular(tempos, por, pc, regra, nomes, gestos, numeros, cmds, **cfg)
                for t, d in disp:
                    if FC.F_rotulo(trechos, fps, t) == "D0X":
                        R[(nv, nc)]["falsos"][d] += 1
                for ini, fim, r in trechos:
                    if r in alvo:
                        a0, b0 = (ini - 1) / fps, (fim - 1) / fps
                        R[(nv, nc)]["certos"][r] += any(a0 <= t <= b0 + 0.5 and d == alvo[r] for t, d in disp)
        print(f"  IPN {k}/{len(videos)} ({time.time() - t0:.0f} s)", flush=True)

    curtos = pickle.load(open(RAIZ / "data/pontos/curtos.pkl", "rb"))
    slovo = [c for c in curtos if c["base"] == "Slovo sem sinal" and c["divisao"] == "test"]
    S = {(v, c): collections.Counter() for v in variantes for c in conjuntos}
    min_slovo = 0.0
    for c in slovo:
        quadros, _ = F.quadros_12(RAIZ / c["caminho"])
        if len(quadros) < FC.QPS:
            continue
        min_slovo += len(quadros) / FC.QPS / 60
        tempos, por, pc, _ = FC.analisar(cnn, None, None, quadros, None, gestos)
        regra = numeros_da_regra(FC.pontos_12(c), tempos)
        for nv, cfg in variantes.items():
            for nc, cmds in conjuntos.items():
                S[(nv, nc)].update(d for _, d in simular(tempos, por, pc, regra, nomes, gestos, numeros, cmds, **cfg))
    print(f"  Slovo pronto ({time.time() - t0:.0f} s)", flush=True)

    saida = {"o_que_e": __doc__.split("\n\n")[0], "ipn_minutos_sem_gesto": round(sem_gesto, 1), "slovo_minutos": round(min_slovo, 1),
             "ipn_trechos_apontar": n_apontar, "variantes": {}}
    for nv in variantes:
        saida["variantes"][nv] = {}
        for nc in conjuntos:
            r, s = R[(nv, nc)], S[(nv, nc)]
            f = sum(r["falsos"].values())
            saida["variantes"][nv][nc] = {
                "ipn_falsos_por_minuto": round(f / sem_gesto, 3), "ipn_falsos": f, "ipn_falsos_por_classe": dict(r["falsos"].most_common()),
                "ipn_um_dedo_trechos_certos": f"{r['certos']['B0A']}/{n_apontar['B0A']}",
                "ipn_dois_dedos_trechos_certos": f"{r['certos']['B0B']}/{n_apontar['B0B']}",
                "slovo_falsos_por_minuto": round(sum(s.values()) / max(min_slovo, 1e-9), 3), "slovo_falsos_por_classe": dict(s.most_common())}
    destino = RAIZ / "resultados/pontos/fluxo_ao_vivo_veto2.json"
    destino.write_text(json.dumps(saida, ensure_ascii=False, indent=1), encoding="utf-8")
    for nv, por_c in saida["variantes"].items():
        for nc, x in por_c.items():
            print(f"== {nv} [{nc}]: IPN {x['ipn_falsos_por_minuto']} falsos/min ({x['ipn_falsos']}) {x['ipn_falsos_por_classe']} | "
                  f"1 dedo {x['ipn_um_dedo_trechos_certos']} | 2 dedos {x['ipn_dois_dedos_trechos_certos']} | Slovo {x['slovo_falsos_por_minuto']}/min")
    print(f"Pronto em {time.time() - t0:.0f} s; gravado em {destino}")


if __name__ == "__main__":
    main()
