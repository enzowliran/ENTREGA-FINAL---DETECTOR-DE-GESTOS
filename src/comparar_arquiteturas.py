"""Junta a comparação de arquiteturas (pastas resultados/jester/arq_*) e mede a latência de cada modelo.

Todas foram treinadas do zero nas mesmas condições: 20 mil vídeos de treino, 8 épocas, lote de 48, seed 42 e
a mesma validação de 5 mil vídeos. A escolha da melhor época usa só a validação; o teste não é usado.

Latência: tempo de uma análise na GPU, com 1 janela de 16 quadros e com 3 janelas (como as demos fazem),
depois de aquecer. Rode com a GPU livre, sem outro treino ao mesmo tempo.

Uso:
    python src/comparar_arquiteturas.py
"""
import csv
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import modelos as M

R = Path("resultados/jester")
NOMES = {"cnn_3d": "CNN 3D", "cnn_3d_leve": "CNN 3D leve", "cnn_3d_larga": "CNN 3D larga", "res_3d": "ResNet 3D",
         "cnn_gru": "CNN 2D + GRU", "cnn_transformer": "CNN 2D + Transformer"}


@torch.no_grad()
def latencia(modelo, janelas, repeticoes=200):
    x = torch.randn(janelas, 16, 3, 96, 96, device="cuda")
    for _ in range(30):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            modelo(x)
    torch.cuda.synchronize()
    inicio = time.perf_counter()
    for _ in range(repeticoes):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            modelo(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - inicio) / repeticoes * 1000


def main():
    torch.backends.cudnn.benchmark = True
    linhas = []
    for arq, nome in NOMES.items():
        pasta = R / f"arq_{arq}"
        if not (pasta / "historico.csv").exists():
            print(f"{nome}: sem resultado, pulando", flush=True)
            continue
        h = list(csv.DictReader((pasta / "historico.csv").open(encoding="utf-8")))
        melhor = max(h, key=lambda l: float(l["top1_val"]))
        config = json.loads((pasta / "config.json").read_text(encoding="utf-8"))
        estado = torch.load(pasta / "melhor.pt", map_location="cuda", weights_only=False)
        modelo = M.criar_modelo(estado["arquitetura"], len(estado["nomes"])).cuda().eval()
        modelo.load_state_dict(estado["modelo"])
        linha = {"arquitetura": arq, "nome": nome, "parametros_mi": round(M.contar_parametros(modelo) / 1e6, 2), "lr": config["lr"],
                 "top1_val": round(100 * float(melhor["top1_val"]), 1), "top5_val": round(100 * float(melhor["top5_val"]), 1),
                 "melhor_epoca": int(melhor["epoca"]) + 1, "falsa_ativacao_06": round(100 * float(melhor["falsa_ativacao_0.6"]), 2),
                 "gestos_acertados_06": round(100 * float(melhor["gestos_acertados_0.6"]), 1),
                 "segundos_epoca": round(sum(float(l["segundos"]) for l in h) / len(h)),
                 "curva_val": [round(100 * float(l["top1_val"]), 1) for l in h],
                 "latencia_1_ms": round(latencia(modelo, 1), 1), "latencia_3_ms": round(latencia(modelo, 3), 1)}
        linhas.append(linha)
        print(f"{nome:22s} {linha['top1_val']:5.1f}%  {linha['parametros_mi']:5.2f} M  {linha['latencia_1_ms']:5.1f} ms  "
              f"{linha['segundos_epoca']} s/época  falsa ativação {linha['falsa_ativacao_06']}%", flush=True)
        del modelo, estado
        torch.cuda.empty_cache()
    linhas.sort(key=lambda l: -l["top1_val"])
    saida = {"protocolo": "20 mil vídeos de treino, 8 épocas, lote 48, seed 42, validação de 5 mil vídeos; sem aumento de dados",
             "fora": {"r2plus1d": "com lote de 48 passa dos 8 GB da GPU e fica dezenas de vezes mais lenta"}, "resultados": linhas}
    (R / "comparacao_arquiteturas.json").write_text(json.dumps(saida, indent=2, ensure_ascii=False), encoding="utf-8")
    with (R / "comparacao_arquiteturas.csv").open("w", newline="", encoding="utf-8") as f:
        campos = [k for k in linhas[0] if k != "curva_val"]
        e = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        e.writeheader()
        e.writerows(linhas)


if __name__ == "__main__":
    main()
