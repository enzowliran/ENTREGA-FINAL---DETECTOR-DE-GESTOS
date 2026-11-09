"""Mede um modelo nos vídeos de demo/gravar_gestos.py, com o gatilho de 0,7 e 3 janelas seguidas.

Cada vídeo só contém um gesto, mantido do início ao fim. Em um vídeo com gesto, conta se a classe certa
disparou e quantos disparos foram de outra classe. Nos vídeos sem gesto, todo disparo é falso.

Uso:
    python src/avaliar_webcam.py --modelo resultados/jester/cnn_3d_mais_gestos/melhor.pt
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import calibrar_fluxo as C
import calibrar_por_gesto as P
import modelos as M
import treinar_jester as T

ALVOS = {"one_finger": "One Finger", "two_fingers": "Two Fingers", "three_fingers": "Three Fingers", "four_fingers": "Four Fingers",
         "fist": "Fist", "stop_sign": "Stop Sign", "thumb_up": "Thumb Up", "thumb_down": "Thumb Down", "sem_gesto": None}
NOVOS_META = 0.70


def ler_video(caminho, tamanho=96):
    """Frames RGB uint8 (T, 3, 96, 96), com o mesmo recorte central da demo."""
    captura, frames = cv2.VideoCapture(str(caminho)), []
    while True:
        ok, imagem = captura.read()
        if not ok:
            break
        a, l = imagem.shape[:2]
        lado = min(a, l)
        y0, x0 = (a - lado) // 2, (l - lado) // 2
        frames.append(cv2.resize(imagem[y0:y0 + lado, x0:x0 + lado], (tamanho, tamanho))[..., ::-1].transpose(2, 0, 1))
    captura.release()
    return np.stack(frames) if frames else np.zeros((0, 3, tamanho, tamanho), dtype=np.uint8)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--modelo", required=True)
    a.add_argument("--pasta", type=Path, default=Path("data/webcam"))
    a.add_argument("--limiar", type=float, default=P.LIMIAR_BASE)
    a.add_argument("--seguidas", type=int, default=P.N_BASE)
    args = a.parse_args()
    estado = torch.load(args.modelo, map_location="cuda", weights_only=False)
    nomes = list(estado["nomes"])
    modelo = M.criar_modelo(estado["arquitetura"], len(nomes)).cuda().eval()
    modelo.load_state_dict(estado["modelo"])
    negativos = [nomes.index(n) for n in T.SEM_GESTO]
    limiares = np.full(len(nomes), np.inf)
    limiares[[i for i in range(len(nomes)) if i not in negativos]] = args.limiar

    linhas = []
    for arquivo in sorted(args.pasta.glob("*/*.mp4")):
        if arquivo.name.endswith(".tmp.mp4"):
            continue
        rotulo = re.sub(r"_\d+$", "", arquivo.stem)
        if rotulo not in ALVOS:
            continue
        alvo = ALVOS[rotulo]
        if alvo is not None and alvo not in nomes:
            continue
        frames = ler_video(arquivo)
        if len(frames) < 40:
            print(f"Vídeo curto demais, ignorado: {arquivo}", flush=True)
            continue
        p, _ = C.prever_fluxo(modelo, frames)
        classe, conf = P.resumir(p, negativos)
        _, disparadas = P.disparos(classe, conf, limiares, args.seguidas)
        disparadas = [nomes[int(c)] for c in disparadas]
        certos = sum(d == alvo for d in disparadas)
        linhas.append({"pessoa": arquivo.parent.name, "gesto": rotulo, "video": arquivo.stem, "segundos": len(frames) / C.FPS_ORIGEM,
                       "detectou": int(certos > 0), "disparos_errados": len(disparadas) - certos, "disparadas": "; ".join(disparadas),
                       "mais_prevista": nomes[int(np.bincount(p.argmax(1), minlength=len(nomes)).argmax())]})
    if not linhas:
        raise SystemExit(f"Nenhum vídeo em {args.pasta}. Grave com: python demo/gravar_gestos.py --pessoa <nome>")

    resumo = {"modelo": args.modelo, "limiar": args.limiar, "seguidas": args.seguidas, "pessoas": sorted({l["pessoa"] for l in linhas}),
              "por_gesto": {}}
    for rotulo in ALVOS:
        parte = [l for l in linhas if l["gesto"] == rotulo]
        if parte and rotulo != "sem_gesto":
            resumo["por_gesto"][rotulo] = {"videos": len(parte), "detectados": sum(l["detectou"] for l in parte),
                                           "disparos_errados": sum(l["disparos_errados"] for l in parte)}
            print(f"{rotulo:15s} {sum(l['detectou'] for l in parte)}/{len(parte)} detectados, {sum(l['disparos_errados'] for l in parte)} disparos errados", flush=True)
    neg = [l for l in linhas if l["gesto"] == "sem_gesto"]
    if neg:
        minutos = sum(l["segundos"] for l in neg) / 60
        falsos = sum(l["disparos_errados"] for l in neg)
        resumo["sem_gesto"] = {"minutos": minutos, "falsos": falsos, "falsos_por_min": falsos / minutos}
        print(f"Sem gesto: {falsos} falsos em {minutos:.1f} min ({falsos / minutos:.2f}/min)", flush=True)
    novos = [l for l in linhas if ALVOS[l["gesto"]] in ("One Finger", "Two Fingers", "Three Fingers", "Four Fingers", "Fist")]
    if novos:
        taxa = sum(l["detectou"] for l in novos) / len(novos)
        resumo["novos_detectados"] = taxa
        print(f"Gestos novos: {taxa:.1%} detectados em {len(novos)} vídeos (critério: pelo menos {NOVOS_META:.0%})", flush=True)

    pasta = Path(args.modelo).parent
    with (pasta / "webcam.csv").open("w", newline="", encoding="utf-8") as f:
        e = csv.DictWriter(f, fieldnames=list(linhas[0]))
        e.writeheader()
        e.writerows(linhas)
    (pasta / "webcam.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
