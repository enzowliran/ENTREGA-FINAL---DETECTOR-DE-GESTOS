"""Mede falsos disparos por minuto em um fluxo simulado com vídeos do Jester.

Os vídeos sem gesto da validação são colados em sequência, a 12 fps, e uma janela de 16 frames
desliza sobre eles como na demo. Um disparo exige o mesmo gesto, com confiança >= limiar, em N
janelas seguidas, com 1,5 s de pausa depois de cada disparo. Os vídeos com gesto medem quantos
gestos seriam detectados com a classe certa. É uma aproximação: os vídeos são independentes, não
uma gravação contínua real.

Uso:
    python src/calibrar_fluxo.py --modelo resultados/jester/cnn_3d_pequeno/melhor.pt
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import jester as J
import modelos as M
import treinar_jester as T

FPS_ORIGEM = 12
PASSO_JANELA = 3  # frames a 12 fps = 0,25 s
INTERVALO_AMOSTRA = 12 / 5.3  # distância entre frames da janela: mesmo ritmo do treino
LIMIARES = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
N_SEGUIDAS = (1, 2, 3, 4, 6)
PAUSA_JANELAS = 6  # 1,5 s


def decodificar(conjunto, posicoes, tamanho=96):
    """Devolve os frames RGB uint8 (T, 3, 96, 96) dos vídeos pedidos, em sequência, e os inícios."""
    pedacos, inicios, total = [], [], 0
    for posicao in posicoes:
        _, inicio, contagem, _ = conjunto.amostras[posicao]
        for i in range(inicio, inicio + contagem):
            d, t = conjunto.deslocamento[i], conjunto.tamanho_jpg[i]
            imagem = cv2.imdecode(np.asarray(conjunto.mapa[d:d + t]), cv2.IMREAD_COLOR)
            a, l = imagem.shape[:2]
            lado = min(a, l)
            y0, x0 = (a - lado) // 2, (l - lado) // 2
            imagem = cv2.resize(imagem[y0:y0 + lado, x0:x0 + lado], (tamanho, tamanho))
            pedacos.append(imagem[..., ::-1].transpose(2, 0, 1))
        inicios.append(total)
        total += contagem
    return np.stack(pedacos), np.array(inicios)


@torch.no_grad()
def prever_fluxo(modelo, frames, lote=64):
    """Probabilidades de todas as janelas deslizantes."""
    offsets = np.round(np.arange(16) * INTERVALO_AMOSTRA).astype(int)
    pontos = np.arange(0, len(frames) - offsets[-1], PASSO_JANELA)
    saida = []
    for i in range(0, len(pontos), lote):
        indices = pontos[i:i + lote, None] + offsets
        x = torch.from_numpy(frames[indices].astype(np.float32)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            saida.append(modelo((x - 127.5) / 127.5).float().softmax(1).cpu())
    return torch.cat(saida).numpy(), pontos


def contar_disparos(p, negativos, limiar, n_seguidas):
    """Devolve (posição da janela, classe) de cada disparo."""
    gesto = p.copy()
    gesto[:, negativos] = 0
    classe, confianca = gesto.argmax(1), gesto.max(1)
    disparos, seguidas, atual, bloqueio = [], 0, -1, -1
    for i in range(len(p)):
        if confianca[i] >= limiar:
            seguidas = seguidas + 1 if classe[i] == atual else 1
            atual = classe[i]
        else:
            seguidas, atual = 0, -1
        if seguidas >= n_seguidas and i > bloqueio:
            disparos.append((i, int(classe[i])))
            bloqueio = i + PAUSA_JANELAS
            seguidas = 0
    return disparos


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--modelo", required=True)
    a.add_argument("--dados", default="data/Jester")
    a.add_argument("--n-gestos", type=int, default=1500, help="Vídeos com gesto usados para medir a detecção")
    args = a.parse_args()
    estado = torch.load(args.modelo, map_location="cuda", weights_only=False)
    nomes = estado["nomes"]
    modelo = M.criar_modelo(estado["arquitetura"], len(nomes)).cuda().eval()
    modelo.load_state_dict(estado["modelo"])
    negativos = [nomes.index(n) for n in T.SEM_GESTO]
    _, _, validacao, _ = J.ler_rotulos(Path(args.dados) / "rotulos/jester_labels")
    conjunto = J.ConjuntoJester(args.dados, validacao, treino=False)

    sem_gesto = [i for i, s in enumerate(conjunto.amostras) if s[3] in negativos]
    com_gesto = [i for i, s in enumerate(conjunto.amostras) if s[3] not in negativos][:args.n_gestos]
    frames, _ = decodificar(conjunto, sem_gesto)
    minutos = len(frames) / FPS_ORIGEM / 60
    p_sem, _ = prever_fluxo(modelo, frames)
    del frames
    print(f"Fluxo sem gesto: {len(sem_gesto)} vídeos, {minutos:.1f} min, {len(p_sem)} janelas", flush=True)

    frames, inicios = decodificar(conjunto, com_gesto)
    p_com, pontos = prever_fluxo(modelo, frames)
    fins = np.r_[inicios[1:], len(frames)]
    alvo = np.array([conjunto.amostras[i][3] for i in com_gesto])
    janela_video = np.searchsorted(inicios, pontos + 18, side="right") - 1

    linhas = []
    for limiar in LIMIARES:
        for n in N_SEGUIDAS:
            falsos = contar_disparos(p_sem, negativos, limiar, n)
            disparos = contar_disparos(p_com, negativos, limiar, n)
            certos = {int(janela_video[i]) for i, c in disparos if c == alvo[janela_video[i]]}
            errados = sum(c != alvo[janela_video[i]] for i, c in disparos)
            linhas.append({"limiar": limiar, "janelas_seguidas": n, "falsos_por_min": len(falsos) / minutos,
                           "falsos": len(falsos), "gestos_detectados": len(certos) / len(com_gesto),
                           "disparos_classe_errada": errados / len(com_gesto),
                           "latencia_min_s": round(n * PASSO_JANELA / FPS_ORIGEM, 2)})
    pasta = Path(args.modelo).parent
    with (pasta / "calibracao_fluxo.csv").open("w", newline="", encoding="utf-8") as f:
        e = csv.DictWriter(f, fieldnames=list(linhas[0]))
        e.writeheader(); e.writerows(linhas)
    print("limiar n  falsos/min falsos detectados errados")
    for l in linhas:
        print(f"{l['limiar']:5.2f} {l['janelas_seguidas']:2d}  {l['falsos_por_min']:9.3f} {l['falsos']:6d} "
              f"{l['gestos_detectados']:9.3f} {l['disparos_classe_errada']:7.3f}", flush=True)
    (pasta / "calibracao_resumo.json").write_text(json.dumps({"minutos": minutos, "n_sem_gesto": len(sem_gesto), "n_gestos": len(com_gesto)}), encoding="utf-8")


if __name__ == "__main__":
    main()
