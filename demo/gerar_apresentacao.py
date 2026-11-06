"""Gera demo/apresentacao.html com os números lidos de resultados/, sem digitar nada à mão.

Uso:
    python demo/gerar_apresentacao.py
"""
import csv
import json
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
R = RAIZ / "resultados/jester"
EQUIVALENTES = {
    "SwipeLeft": ("Swiping Left", "Sliding Two Fingers Left"), "SwipeRight": ("Swiping Right", "Sliding Two Fingers Right"),
    "FlickUp": ("Swiping Up", "Sliding Two Fingers Up"), "FlickDown": ("Swiping Down", "Sliding Two Fingers Down"),
}


def historico(pasta):
    with (R / pasta / "historico.csv").open(encoding="utf-8") as f:
        return [round(float(l["top1_val"]) * 100, 1) for l in csv.DictReader(f)]


def linhas_fluxo(pasta):
    with (R / pasta / "calibracao_fluxo.csv").open(encoding="utf-8") as f:
        return [{"limiar": float(l["limiar"]), "janelas": int(l["janelas_seguidas"]), "fp": float(l["falsos_por_min"]),
                 "falsos": int(l["falsos"]), "det": float(l["gestos_detectados"]), "err": float(l["disparos_classe_errada"])}
                for l in csv.DictReader(f)]


def fronteira(linhas):
    """Para cada taxa de falsos disparos, fica o gatilho que detecta mais gestos."""
    melhores = []
    for l in sorted(linhas, key=lambda x: (x["fp"], -x["det"])):
        if not melhores or l["det"] * 100 > melhores[-1]["det"]:
            melhores.append({"fp": round(l["fp"], 4), "det": round(l["det"] * 100, 1)})
    return melhores


def pct(x, casas=1):
    return f"{float(x) * 100:.{casas}f}".replace(".", ",") + "%"


def sem_ajuste():
    arquivo = RAIZ / "resultados/drivermhg/jester_sem_ajuste.csv"
    with arquivo.open(encoding="utf-8") as f:
        linhas = [l for l in csv.DictReader(f) if l["fonte"] == "rgb" and l["modo"] == "normal" and l["gesto"] in EQUIVALENTES]
    return sum(l["predito"] in EQUIVALENTES[l["gesto"]] for l in linhas), len(linhas)


teste = json.loads((R / "cnn_3d/teste.json").read_text(encoding="utf-8"))
fluxo = linhas_fluxo("cnn_3d")
escolhido = next(l for l in fluxo if l["limiar"] == 0.7 and l["janelas"] == 3)
acertos_volante, total_volante = sem_ajuste()

valores = {
    "top1": pct(teste["top1"]), "top5": pct(teste["top5"]), "n_teste": f"{teste['n']:,}".replace(",", "."),
    "fp": f"{escolhido['fp']:.2f}".replace(".", ","), "falsos": str(escolhido["falsos"]), "det": pct(escolhido["det"]),
    "acertos_volante": str(acertos_volante), "total_volante": str(total_volante),
}
dados = {
    "historico": {"Pequeno": historico("cnn_3d_pequeno"), "Médio": historico("cnn_3d_medio"), "Grande": historico("cnn_3d")},
    "fronteira": {"Pequeno": fronteira(linhas_fluxo("cnn_3d_pequeno")), "Médio": fronteira(linhas_fluxo("cnn_3d_medio")), "Grande": fronteira(fluxo)},
    "escolhido": {"fp": escolhido["fp"], "det": round(escolhido["det"] * 100, 1)},
    "top1": round(teste["top1"] * 100, 1),
    "volante": {"acertos": acertos_volante, "total": total_volante},
}

modelo = (RAIZ / "demo/apresentacao_modelo.html").read_text(encoding="utf-8")
for chave, valor in valores.items():
    modelo = modelo.replace("{{" + chave + "}}", valor)
modelo = modelo.replace("/*DADOS*/null", json.dumps(dados, ensure_ascii=False))
if "{{" in modelo:
    raise ValueError("Marcador sem valor: " + modelo[modelo.index("{{"):][:30])
(RAIZ / "demo/apresentacao.html").write_text(modelo, encoding="utf-8")
print("Gerado demo/apresentacao.html;", valores["top1"], "no teste;", valores["fp"], "falsos/min;", valores["det"], "detectados;",
      acertos_volante, "de", total_volante, "no volante")
