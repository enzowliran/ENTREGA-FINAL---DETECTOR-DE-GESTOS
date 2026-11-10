"""Passa um vídeo sintético (uma mão desenhada deslizando para a direita) pela CNN 3D treinada e guarda as ativações.

Serve à apresentação: os mapas mostrados nos slides da CNN 3D são saídas reais do modelo, sem usar quadros do
Jester, que a licença não permite publicar. De cada bloco, guarda os 8 mapas que mais variam no instante do meio,
normalizados de 0 a 255, além do vetor de 256 números depois da média e das probabilidades dos 27 gestos.

Uso:
    python src/ativacoes_exemplo.py
"""
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import modelos as M


def video_sintetico(n=16, lado=96):
    """Fundo claro com textura leve e uma mão cor de pele que atravessa o quadro da esquerda para a direita."""
    gerador = np.random.default_rng(7)
    quadros = []
    for i in range(n):
        q = np.full((lado, lado, 3), (214, 220, 226), np.uint8)
        q = (q.astype(np.int16) + gerador.integers(-6, 7, q.shape)).clip(0, 255).astype(np.uint8)
        cv2.rectangle(q, (0, 70), (lado, lado), (120, 130, 150), -1)  # ombro e roupa, parados
        cx = int(18 + (lado - 36) * i / (n - 1))
        pele = (200, 150, 120)
        cv2.ellipse(q, (cx, 52), (13, 15), 0, 0, 360, pele, -1)
        for d in range(4):
            cv2.rectangle(q, (cx - 11 + d * 6, 24 + (d % 2) * 3), (cx - 7 + d * 6, 46), pele, -1)
        cv2.rectangle(q, (cx - 17, 46), (cx - 12, 58), pele, -1)  # polegar
        cv2.rectangle(q, (cx - 7, 62), (cx + 7, 80), (190, 140, 110), -1)  # punho
        quadros.append(cv2.GaussianBlur(q, (3, 3), 0))
    return np.stack(quadros)


def para_bytes(mapa):
    m = mapa.astype(np.float32)
    m = (m - m.min()) / (m.max() - m.min() + 1e-6)
    return base64.b64encode((m * 255).astype(np.uint8).tobytes()).decode()


@torch.no_grad()
def main():
    estado = torch.load("resultados/jester/cnn_3d/melhor.pt", map_location="cpu", weights_only=False)
    nomes = list(estado["nomes"])
    modelo = M.criar_modelo(estado["arquitetura"], len(nomes)).eval()
    modelo.load_state_dict(estado["modelo"])
    video = video_sintetico()
    x = (torch.from_numpy(video).permute(0, 3, 1, 2).float()[None] - 127.5) / 127.5  # (1, T, C, H, W), RGB, como no treino
    saidas = []
    ganchos = [m.register_forward_hook(lambda _m, _e, s: saidas.append(s.detach())) for m in modelo.corpo if isinstance(m, torch.nn.MaxPool3d)]
    logits = modelo(x)
    for g in ganchos:
        g.remove()
    blocos = []
    for i, s in enumerate(saidas):
        s = s[0]  # (C, T, H, W)
        meio = s[:, s.shape[1] // 2]
        escolhidos = meio.flatten(1).std(1).argsort(descending=True)[:8].tolist()
        blocos.append({"bloco": i + 1, "canais": int(s.shape[0]), "T": int(s.shape[1]), "lado": int(s.shape[2]), "mapas": [para_bytes(meio[c].numpy()) for c in escolhidos],
                       "media_por_canal": [round(float(v), 4) for v in s.mean((1, 2, 3))[:64]]})
    vetor = torch.nn.functional.adaptive_avg_pool3d(saidas[-1], 1).flatten()  # o mesmo que o modelo faz antes da última camada
    prob = logits.softmax(1)[0]
    quadros_mostrados = [base64.b64encode(cv2.resize(video[i], (48, 48), interpolation=cv2.INTER_AREA).tobytes()).decode() for i in range(0, 16, 2)]
    saida = {"descricao": "Ativações reais da CNN 3D (cnn_3d/melhor.pt) para um vídeo sintético de uma mão deslizando para a direita; não usa quadros do Jester.",
             "quadros": quadros_mostrados, "lado_quadro": 48, "blocos": blocos,
             "vetor_256": [round(float(v), 4) for v in vetor],
             "probabilidades": sorted([{"gesto": nomes[i], "p": round(float(prob[i]), 4)} for i in range(len(nomes))], key=lambda d: -d["p"])}
    Path("resultados/jester/cnn_3d/ativacoes_exemplo.json").write_text(json.dumps(saida, ensure_ascii=False), encoding="utf-8")
    print("formas:", [(b["canais"], b["T"], b["lado"]) for b in blocos])
    print("3 gestos mais prováveis:", saida["probabilidades"][:3])


if __name__ == "__main__":
    main()
