"""Quanto a janela do OK reduz os falsos disparos de número, em vídeo contínuo (simulação, sem rodar rede).

Por quê: sem a janela do OK, o modelo final dispara cerca de 1 comando do carro por minuto nos vídeos do IPN sem
gesto, e todos esses falsos são números da contagem de dedos (src/fluxo_ao_vivo.py). No carro, um gesto só vira
comando nos 7 s depois do sinal de OK. Aqui a mesma contagem de dedos roda nos mesmos vídeos, e o detector do OK
(o de demo/carro3d.html, portado para Python) decide quando a janela está aberta.

Só pontos da mão (data/pontos/ipn.pkl, MediaPipe já extraído), CPU:
  - números: contagem de dedos com a mão parada, mesmo número em 5 de 6 quadros (src/medir_confianca_numeros.py),
    uma análise a cada 0,15 s, pausa de 1,2 s depois de um disparo, só os números do carro (1, 2 e 4 dedos);
  - OK: pinça (polegar-indicador) < 0,5 palma e os três outros dedos retos (> 0,78), mantido por 0,4 s, com
    tolerância de 0,25 s a falhas; precisa soltar o OK entre duas ativações; a janela dura 7 s.
Limites da simulação: pontos a 12 quadros por segundo e só a mão principal (no navegador, o OK roda na taxa da
câmera e olha até duas mãos); a CNN não entra (nos mesmos vídeos ela não deu nenhum falso de comando do carro).

Uso:
    demo\\python_pontos.bat src\\fluxo_janela_ok.py
"""
import collections
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fluxo_combinado as FC
import medir_confianca_numeros as MC

RAIZ = Path(__file__).resolve().parent.parent
NUMEROS_CARRO = {"One Finger", "Two Fingers", "Four Fingers"}
PAUSA, SOLTAR = 1.2, 0.5
OK_PINCA, OK_RETA, OK_SEGURAR, OK_TOLERA, JANELA_S = 0.5, 0.78, 0.4, 0.25, 7.0


def e_ok(w, l, largura, altura):
    """O teste de demo/carro3d.html (medidasOk) num quadro: w = pontos 3D (metros), l = pontos na imagem (0 a 1)."""
    p2 = l[:, :2] * np.array([largura, altura], np.float32)

    def d3(i, j):
        return float(np.linalg.norm(w[i] - w[j]))

    def d2(i, j):
        return float(np.linalg.norm(p2[i] - p2[j]))

    def reta(d, a, b):
        return d(a, b) / max(d(a, a + 1) + d(a + 1, a + 2) + d(a + 2, b), 1e-6)
    pinca = min(d3(4, 8) / max(d3(0, 9), 1e-6), d2(4, 8) / max(d2(0, 9), 1e-6))
    dedos = [max(reta(d3, a, b), reta(d2, a, b)) for a, b in ((9, 12), (13, 16), (17, 20))]
    return pinca < OK_PINCA and all(r > OK_RETA for r in dedos)


def janelas_do_ok(pts):
    """Instantes (s) em que a janela abriria, pela mesma máquina de estados da página."""
    img, world, pres = pts["img"][:, 0], pts["world"][:, 0], pts["present"][:, 0]
    altura, largura = pts["altura_largura"]
    aberturas, ok_desde, ok_solto, ultimo_ok, ate = [], None, True, -1e9, -1e9
    for f in range(len(pres)):
        agora = f / FC.QPS
        algum = bool(pres[f]) and e_ok(world[f], img[f], largura, altura)
        if algum:
            ultimo_ok = agora
        if not algum and agora - ultimo_ok > OK_TOLERA:
            ok_desde, ok_solto = None, True
            continue
        if not algum:
            continue
        if ok_desde is None:
            ok_desde = agora
        if ok_solto and agora >= ate and agora - ok_desde >= OK_SEGURAR:
            ok_solto, ate = False, agora + JANELA_S
            aberturas.append(agora)
    return aberturas


def disparos_de_numero(pts, n_quadros):
    """(instante, número) de cada disparo da contagem de dedos, com o gatilho do player (uma vez enquanto mantido)."""
    img, world, pres, al = pts["img"][:, 0], pts["world"][:, 0], pts["present"][:, 0], pts["altura_largura"]
    passo, t = FC.INTERVALO * FC.QPS, min(int(round(j * FC.QPS)) for j in FC.JANELAS)
    saida, nome_ant, visto, disparou, fim_pausa = [], None, -1e9, None, 0.0
    while t <= n_quadros:
        agora = t / FC.QPS
        f = min(t, len(pres)) - 1
        n = None
        if f >= MC.N - 1 and MC.parada(img, pres, al, f):
            n = MC.contagem(world, pres, f)
        if n in MC.NUMEROS:
            mantido = n == nome_ant and agora - visto < SOLTAR
            nome_ant, visto = n, agora
            if not mantido:
                disparou = None
            if n in NUMEROS_CARRO and disparou != n and agora >= fim_pausa:
                disparou, fim_pausa = n, agora + PAUSA
                saida.append((agora, n))
        t = int(round(t + passo))
    return saida


def main():
    ipn = pickle.load(open(RAIZ / "data/pontos/ipn.pkl", "rb"))
    videos = sorted(v for v, r in ipn.items() if r["divisao"] == "test")
    minutos, sem, com, aberturas_sem_gesto, aberturas_total = 0.0, collections.Counter(), collections.Counter(), 0, 0
    for v in videos:
        reg = ipn[v]
        cap = cv2.VideoCapture(str(RAIZ / f"data/IPN/videos/videos/{v}.avi"))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_quadros = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps * FC.QPS))
        cap.release()
        pts, trechos = FC.pontos_12(reg), reg["trechos"]
        minutos += sum(f - i + 1 for i, f, r in trechos if r == "D0X") / fps / 60
        abre = janelas_do_ok(pts)
        aberturas_total += len(abre)
        aberturas_sem_gesto += sum(FC.F_rotulo(trechos, fps, a) == "D0X" for a in abre)
        for t, n in disparos_de_numero(pts, n_quadros):
            if FC.F_rotulo(trechos, fps, t) != "D0X":
                continue
            sem[n] += 1
            if any(a <= t <= a + JANELA_S for a in abre):
                com[n] += 1
    r = {"o_que_e": __doc__.split("\n\n")[0],
         "limites": "simulação com os pontos da mão a 12 quadros/s e só a mão principal; a CNN não entra",
         "videos": len(videos), "ipn_minutos_sem_gesto": round(minutos, 1),
         "sem_janela": {"falsos": sum(sem.values()), "falsos_por_minuto": round(sum(sem.values()) / minutos, 3), "por_classe": dict(sem)},
         "com_janela_do_ok": {"falsos": sum(com.values()), "falsos_por_minuto": round(sum(com.values()) / minutos, 3), "por_classe": dict(com)},
         "ok_abriu_sozinho": {"vezes_em_trechos_sem_gesto": aberturas_sem_gesto,
                              "por_minuto": round(aberturas_sem_gesto / minutos, 3), "vezes_no_video_todo": aberturas_total}}
    (RAIZ / "resultados/pontos/fluxo_janela_ok.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
