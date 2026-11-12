"""Mede os números pelos pontos da mão (src/pontos_mao.py) em vídeo real, como se fosse a webcam.

Mesmo critério do src/fluxo_ipn.py, para dar para comparar com as CNNs:
  - IPN Hand, 52 vídeos de validação (13 pessoas): falsos disparos por minuto nos trechos sem gesto (D0X),
    com 1,5 s de tolerância logo depois de um gesto; e, nos trechos de apontar com 1 e 2 dedos (B0A e B0B),
    quantos trechos tiveram o número certo disparado.
  - Vídeos de sinais "um" e "dois" (Slovo e ASL Citizen): quantos vídeos dispararam o número certo.
  - Slovo no_event (a pessoa parada, sem sinal): falsos disparos por minuto.

Além dos disparos, conta quadro a quadro: em quantos o MediaPipe achou a mão e em quantos a pose lida
foi a certa. Isso separa "a forma da mão foi lida certo" de "o decisor disparou".

Os quadros entram inteiros (não recortados), a `--qps` quadros por segundo, com o tempo do próprio vídeo.
Só precisa do MediaPipe (roda com demo\\python_pontos.bat), não usa a CNN.

Uso:
    demo\\python_pontos.bat src\\medir_pontos_videos.py
"""
import argparse
import collections
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
import pontos_mao as P

RAIZ = Path(__file__).resolve().parent.parent
ALVO = {"B0A": "One Finger", "B0B": "Two Fingers"}


def rodar_video(caminho, qps, opcoes):
    """Roda detector + decisor no vídeo. Devolve fps original, duração e, por quadro, (t, mão achada, pose), e os disparos."""
    detector = P.DetectorMao(video=True)
    decisor = P.DecisorNumeros(**opcoes)
    cap = cv2.VideoCapture(str(caminho))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    quadros, disparos, i, proximo = [], [], 0, 0.0
    while True:
        ok, q = cap.read()
        if not ok:
            break
        if i >= proximo:
            t = i / fps
            mao = detector.detectar(q, agora=t)
            altura, largura = q.shape[:2]
            numero = decisor.atualizar(mao, t, largura, altura)
            quadros.append((t, mao is not None, decisor.pose))
            if numero:
                disparos.append((t, numero))
            proximo += fps / qps
        i += 1
    cap.release()
    detector.fechar()
    return fps, i / fps, quadros, disparos


def tarefa_ipn(args):
    video, trechos, qps, opcoes = args
    fps, duracao, quadros, disparos = rodar_video(RAIZ / f"data/IPN/videos/videos/{video}.avi", qps, opcoes)

    def rotulo(seg):  # igual ao fluxo_ipn.py: anotação em quadros de 30 fps, começando em 1
        f = seg * fps + 1
        for ini, fim, r in trechos:
            if ini <= f <= fim:
                if r == "D0X" and any(x != "D0X" and fim2 < f <= fim2 + 1.5 * fps for _, fim2, x in trechos):
                    return "logo após um gesto"
                return r
        return None

    R = {"minutos": duracao / 60, "minutos_sem_gesto": sum(f - i + 1 for i, f, r in trechos if r == "D0X") / fps / 60,
         "disparos_por_trecho": collections.Counter(), "falsos": collections.Counter(),
         "apontar": {r: {"trechos": 0, "com_certo": 0, "com_algum": 0, "disparos": collections.Counter(),
                         "quadros": 0, "quadros_mao": 0, "quadros_pose_certa": 0} for r in ALVO}}
    for t, numero in disparos:
        r = rotulo(t)
        R["disparos_por_trecho"][r or "entre trechos"] += 1
        if r == "D0X":
            R["falsos"][numero] += 1
    for ini, fim, r in trechos:
        if r not in ALVO:
            continue
        A = R["apontar"][r]
        a, b = (ini - 1) / fps, (fim - 1) / fps
        nums = [n for t, n in disparos if a <= t <= b + 0.5]  # meio segundo de folga no fim do trecho
        A["trechos"] += 1
        A["com_algum"] += bool(nums)
        A["com_certo"] += ALVO[r] in nums
        A["disparos"].update(nums)
        for t, achou, pose in quadros:
            if a <= t <= b:
                A["quadros"] += 1
                A["quadros_mao"] += achou
                A["quadros_pose_certa"] += pose == ALVO[r]
    return video, R


def tarefa_simples(args):
    base, caminho, classe, qps, opcoes, inicio, fim = args
    fps, duracao, quadros, disparos = rodar_video(caminho, qps, opcoes)
    if inicio is not None and fim and fim > inicio:  # só o sinal, com 0,5 s de folga dos dois lados
        a, b = inicio / fps - 0.5, fim / fps + 0.5
        quadros = [q for q in quadros if a <= q[0] <= b]
        disparos = [d for d in disparos if a <= d[0] <= b + 0.5]
    return {"base": base, "video": Path(caminho).stem, "classe": classe, "minutos": duracao / 60,
            "disparos": [n for _, n in disparos], "quadros": len(quadros),
            "quadros_mao": sum(q[1] for q in quadros), "quadros_pose_certa": sum(q[2] == classe for q in quadros)}


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--qps", type=float, default=15.0, help="quadros por segundo analisados (a webcam com o MediaPipe roda a ~16-20)")
    a.add_argument("--tempo-estavel", type=float, default=0.5)
    a.add_argument("--processos", type=int, default=6)
    a.add_argument("--saida", default="resultados/pontos/videos_reais.json")
    args = a.parse_args()
    opcoes = {"tempo_estavel": args.tempo_estavel}

    meta = {l["Video Name"]: l["Set"] for l in csv.DictReader(open(RAIZ / "data/IPN/anotacoes/metadata.csv", encoding="utf-8"))}
    anot = collections.defaultdict(list)
    for l in csv.DictReader(open(RAIZ / "data/IPN/anotacoes/Annot_List.txt", encoding="utf-8")):
        anot[l["video"]].append((int(l["t_start"]), int(l["t_end"]), l["label"]))
    videos = sorted(v for v in anot if meta.get(v) == "test")

    simples = []
    for l in csv.DictReader(open(RAIZ / "data/Slovo/annotations.tsv", encoding="utf-8"), delimiter="\t"):
        caminho = RAIZ / f"data/Slovo/videos/{l['attachment_id']}.mp4"
        if not caminho.exists():
            continue
        if l["text"] == "no_event":
            simples.append(("Slovo sem sinal", caminho, "No gesture", args.qps, opcoes, None, None))
        elif l["text"] in ("один", "два"):
            classe = "One Finger" if l["text"] == "один" else "Two Fingers"
            simples.append(("Slovo sinais", caminho, classe, args.qps, opcoes, int(l["begin"]), int(l["end"])))
    for s in ("train", "val", "test"):
        for l in csv.DictReader(open(RAIZ / f"data/ASLCitizen/{s}.csv", encoding="utf-8")):
            caminho = RAIZ / "data/ASLCitizen/videos" / Path(l["Video file"]).name
            if l["Gloss"] in ("ONE", "TWO") and caminho.exists():
                simples.append(("ASL Citizen sinais", caminho, "One Finger" if l["Gloss"] == "ONE" else "Two Fingers",
                                args.qps, opcoes, None, None))
    print(f"{len(videos)} vídeos do IPN e {len(simples)} vídeos curtos (Slovo e ASL Citizen)", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(args.processos) as ex:
        ipn = []
        for k, r in enumerate(ex.map(tarefa_ipn, [(v, anot[v], args.qps, opcoes) for v in videos]), 1):
            ipn.append(r)
            if k % 4 == 0:
                print(f"  IPN {k}/{len(videos)} ({time.time() - t0:.0f} s)", flush=True)
        curtos = list(ex.map(tarefa_simples, simples, chunksize=4))
    print(f"Pronto em {(time.time() - t0) / 60:.1f} min", flush=True)

    # IPN
    minutos = sum(r["minutos"] for _, r in ipn)
    sem_gesto = sum(r["minutos_sem_gesto"] for _, r in ipn)
    falsos = sum((r["falsos"] for _, r in ipn), collections.Counter())
    por_trecho = sum((r["disparos_por_trecho"] for _, r in ipn), collections.Counter())
    saida = {"opcoes": {"qps": args.qps, **opcoes}, "ipn": {
        "videos": len(ipn), "minutos_de_video": round(minutos, 1), "minutos_sem_gesto": round(sem_gesto, 1),
        "falsos_em_sem_gesto": sum(falsos.values()), "falsos_por_minuto_sem_gesto": round(sum(falsos.values()) / sem_gesto, 3),
        "falsos_por_classe": dict(falsos.most_common()), "disparos_por_trecho": dict(por_trecho.most_common())}}
    for r, alvo in ALVO.items():
        soma = collections.Counter()
        disp = collections.Counter()
        for _, R in ipn:
            A = R["apontar"][r]
            soma.update({k: v for k, v in A.items() if k != "disparos"})
            disp.update(A["disparos"])
        saida["ipn"][f"apontar_{alvo}"] = {
            "trechos": soma["trechos"], "trechos_com_disparo_certo": soma["com_certo"],
            "pct_trechos_certos": round(100 * soma["com_certo"] / max(soma["trechos"], 1), 1),
            "trechos_com_algum_disparo": soma["com_algum"], "disparos": dict(disp.most_common()),
            "pct_quadros_com_mao": round(100 * soma["quadros_mao"] / max(soma["quadros"], 1), 1),
            "pct_quadros_pose_certa": round(100 * soma["quadros_pose_certa"] / max(soma["quadros"], 1), 1)}

    # Vídeos curtos
    for base in ("Slovo sinais", "ASL Citizen sinais", "Slovo sem sinal"):
        grupo = [c for c in curtos if c["base"] == base]
        if not grupo:
            continue
        if base == "Slovo sem sinal":
            m = sum(c["minutos"] for c in grupo)
            n = sum(len(c["disparos"]) for c in grupo)
            saida[base] = {"videos": len(grupo), "minutos": round(m, 1), "falsos": n, "falsos_por_minuto": round(n / m, 3),
                           "falsos_por_classe": dict(collections.Counter(d for c in grupo for d in c["disparos"]).most_common()),
                           "pct_quadros_com_mao": round(100 * sum(c["quadros_mao"] for c in grupo) / max(sum(c["quadros"] for c in grupo), 1), 1)}
            continue
        saida[base] = {}
        for classe in ("One Finger", "Two Fingers"):
            g = [c for c in grupo if c["classe"] == classe]
            q = max(sum(c["quadros"] for c in g), 1)
            saida[base][classe] = {
                "videos": len(g), "videos_com_disparo_certo": sum(classe in c["disparos"] for c in g),
                "pct_videos_certos": round(100 * sum(classe in c["disparos"] for c in g) / max(len(g), 1), 1),
                "disparos": dict(collections.Counter(d for c in g for d in c["disparos"]).most_common()),
                "pct_quadros_com_mao": round(100 * sum(c["quadros_mao"] for c in g) / q, 1),
                "pct_quadros_pose_certa": round(100 * sum(c["quadros_pose_certa"] for c in g) / q, 1)}

    destino = RAIZ / args.saida
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(saida, indent=2, ensure_ascii=False))
    print(f"Salvo em {destino}")


if __name__ == "__main__":
    main()
