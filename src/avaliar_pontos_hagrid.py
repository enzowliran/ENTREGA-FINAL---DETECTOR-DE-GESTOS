"""Mede a contagem de dedos por pontos (src/pontos_mao.py) em fotos do HaGRIDv2 e ajusta os limiares.

As fotos são as mesmas do treino dos gestos novos da CNN, mas aqui basta uma amostra pequena da
validação (60 por classe), baixada com:
    python src/baixar_hagrid.py --destino data/HaGRID_amostra --por-classe 0

Cada foto é recortada em volta da mão do gesto (caixa das anotações, com folga) e passa pelo Hand
Landmarker em modo imagem. As pessoas são divididas em duas metades pelo user_id: os limiares são
ajustados numa metade e o acerto é medido na outra, para o número não sair otimista. Grava só
números em resultados/pontos/ (nenhuma imagem).

Uso (no ambiente com o MediaPipe):
    demo\\python_pontos.bat src/avaliar_pontos_hagrid.py
"""
import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pontos_mao as P

RAIZ = Path(__file__).resolve().parent.parent
# O que a contagem deveria dizer para cada classe do HaGRID. None = não é número (polegar sozinho, mão aberta).
ESPERADO = {"one": "One Finger", "peace": "Two Fingers", "two_up": "Two Fingers", "three": "Three Fingers",
            "three2": "Three Fingers", "four": "Four Fingers", "fist": "Fist", "like": None, "dislike": None,
            "stop": None, "palm": None, "call": "Two Fingers", "rock": "Two Fingers", "little_finger": "One Finger"}
# Só informativas: a contagem "certa" depende de como se lê o gesto (ok, mute, point) ou não há gesto.
INFORMATIVAS = ("ok", "mute", "point", "no_gesture")
GRADE = {"reta": np.arange(0.70, 0.951, 0.025), "angulo": np.arange(35, 86, 5.0),
         "polegar_reta": np.arange(0.70, 0.971, 0.025), "polegar_fora": np.arange(0.70, 1.81, 0.05)}


def metade(user_id):
    return int(hashlib.md5(str(user_id).encode()).hexdigest(), 16) % 2


def recortar(imagem, caixa, folga=1.8):
    """Recorte quadrado em volta da caixa [x, y, l, a] (de 0 a 1), com folga para a mão inteira caber."""
    altura, largura = imagem.shape[:2]
    x, y, l, a = caixa
    cx, cy = (x + l / 2) * largura, (y + a / 2) * altura
    lado = max(l * largura, a * altura) * folga
    x0, y0 = int(max(cx - lado / 2, 0)), int(max(cy - lado / 2, 0))
    x1, y1 = int(min(cx + lado / 2, largura)), int(min(cy + lado / 2, altura))
    return imagem[y0:y1, x0:x1]


def extrair(pasta, anotacoes):
    detector = P.DetectorMao(video=False, largura=0)
    linhas = []
    arquivos = sorted((pasta / "val").glob("*/*.jpg"))
    t = time.time()
    for i, arquivo in enumerate(arquivos):
        classe, identificador = arquivo.parent.name, arquivo.stem
        a = anotacoes.get(identificador)
        if a is None:
            continue
        imagem = cv2.imread(str(arquivo))
        if imagem is None:
            continue
        rotulos = a.get("labels") or []
        caixas = a.get("bboxes") or []
        indice = next((k for k, r in enumerate(rotulos) if r == classe), 0 if caixas else None)
        recorte = recortar(imagem, caixas[indice]) if indice is not None else imagem
        mao = detector.detectar(recorte) if recorte.size else None
        linha = {"classe": classe, "metade": metade(a.get("user_id")), "detectou": mao is not None}
        if mao is not None:
            linha.update(P.medidas(mao.mundo))
        linhas.append(linha)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(arquivos)} fotos, {time.time() - t:.0f} s", flush=True)
    detector.fechar()
    return linhas


def prever(linhas, limiares):
    """Mesma regra de pontos_mao.contar_dedos, sobre as medidas já extraídas (para a grade ser rápida)."""
    saida = []
    for l in linhas:
        if not l["detectou"]:
            saida.append("sem_mao")
            continue
        polegar = l["polegar_reta"] >= limiares["polegar_reta"] and l["polegar_fora"] >= limiares["polegar_fora"]
        outros = [l[f"{n}_reta"] >= limiares["reta"] and l[f"{n}_angulo"] <= limiares["angulo"] for n in P.NOMES_DEDOS[1:]]
        saida.append(P.classificar_pose((polegar, *outros)))
    return saida


def acerto_por_classe(linhas, previsoes):
    por = {}
    for l, p in zip(linhas, previsoes):
        if l["classe"] in ESPERADO:
            por.setdefault(l["classe"], []).append(p == ESPERADO[l["classe"]])
    return {c: float(np.mean(v)) for c, v in sorted(por.items())}


def ajustar(linhas):
    """Grade em duas etapas, porque os limiares dos 4 dedos e os do polegar quase não interagem."""
    def nota(lim):
        por = acerto_por_classe(linhas, prever(linhas, lim))
        return float(np.mean(list(por.values())))

    melhor = dict(P.LIMIARES_INICIAIS)
    for chaves in (("reta", "angulo"), ("polegar_reta", "polegar_fora")):
        resultados = []
        for valores in itertools.product(*(GRADE[c] for c in chaves)):
            lim = {**melhor, **{c: float(v) for c, v in zip(chaves, valores)}}
            resultados.append((nota(lim), lim))
        melhor = max(resultados, key=lambda r: r[0])[1]
    return {c: round(v, 3) for c, v in melhor.items()}, nota(melhor)


def resumo(linhas, limiares):
    previsoes = prever(linhas, limiares)
    por = acerto_por_classe(linhas, previsoes)
    numeros = [c for c in por if ESPERADO[c] is not None and ESPERADO[c] in P.NUMEROS and c in ESPERADO
               and c not in ("call", "rock", "little_finger")]
    nao_numeros = [c for c in por if ESPERADO[c] is None]
    detectadas = [l for l in linhas if l["classe"] in ESPERADO]
    confusao = {}
    for l, p in zip(linhas, previsoes):
        confusao.setdefault(l["classe"], {}).setdefault(str(p), 0)
        confusao[l["classe"]][str(p)] += 1
    com_mao = [(l, p) for l, p in zip(linhas, previsoes) if l["detectou"] and l["classe"] in numeros]
    return {
        "fotos": len(detectadas),
        "mao_encontrada": float(np.mean([l["detectou"] for l in detectadas])) if detectadas else 0.0,
        "acerto_medio_numeros": float(np.mean([por[c] for c in numeros])) if numeros else None,
        # Só as fotos em que o MediaPipe achou a mão: separa o erro da contagem do erro de não achar a mão.
        "acerto_numeros_quando_acha_mao": float(np.mean([p == ESPERADO[l["classe"]] for l, p in com_mao])) if com_mao else None,
        "nao_numero_ficou_sem_numero": float(np.mean([por[c] for c in nao_numeros])) if nao_numeros else None,
        "acerto_por_classe": por,
        "confusao": confusao,
    }


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--hagrid", type=Path, default=RAIZ / "data/HaGRID_amostra")
    a.add_argument("--saida", type=Path, default=RAIZ / "resultados/pontos")
    a.add_argument("--reaproveitar", action="store_true", help="Usa as medidas já extraídas, sem rodar o MediaPipe de novo")
    args = a.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)
    medidas_arquivo = args.saida / "hagrid_amostra_medidas.json"
    if args.reaproveitar and medidas_arquivo.exists():
        linhas = json.loads(medidas_arquivo.read_text(encoding="utf-8"))
    else:
        anotacoes = json.loads((args.hagrid / "anotacoes.json").read_text(encoding="utf-8"))
        linhas = extrair(args.hagrid, anotacoes)
        medidas_arquivo.write_text(json.dumps(linhas), encoding="utf-8")

    ajuste = [l for l in linhas if l["metade"] == 0]
    teste = [l for l in linhas if l["metade"] == 1]
    limiares, nota_ajuste = ajustar(ajuste)
    r = {
        "o_que_e": "Contagem de dedos por regras sobre os pontos do MediaPipe, em fotos de validação do HaGRIDv2. "
                   "Limiares ajustados na metade 'ajuste' das pessoas e medidos na metade 'teste'.",
        "limiares_iniciais": P.LIMIARES_INICIAIS,
        "limiares_em_uso_em_pontos_mao": P.LIMIARES,
        "limiares_ajustados": limiares,
        "nota_na_metade_de_ajuste": nota_ajuste,
        "teste_com_limiares_iniciais": resumo(teste, P.LIMIARES_INICIAIS),
        "teste_com_limiares_ajustados": resumo(teste, limiares),
        "informativas_com_limiares_ajustados": {c: v for c, v in resumo(linhas, limiares)["confusao"].items() if c in INFORMATIVAS},
        "esperado": ESPERADO,
    }
    (args.saida / "hagrid_amostra.json").write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    for nome in ("teste_com_limiares_iniciais", "teste_com_limiares_ajustados"):
        x = r[nome]
        print(f"\n{nome}: {x['fotos']} fotos, mão encontrada em {x['mao_encontrada']:.0%}")
        print(f"  acerto médio nos números (um a quatro, punho): {x['acerto_medio_numeros']:.1%}"
              f" (só fotos em que achou a mão: {x['acerto_numeros_quando_acha_mao']:.1%})")
        print(f"  joinha/pare sem virar número: {x['nao_numero_ficou_sem_numero']:.1%}")
        for c, v in x["acerto_por_classe"].items():
            print(f"    {c:14s} {v:6.1%}")
    print("\nLimiares ajustados:", limiares)
    print(f"Salvo em {args.saida / 'hagrid_amostra.json'}")


if __name__ == "__main__":
    main()
