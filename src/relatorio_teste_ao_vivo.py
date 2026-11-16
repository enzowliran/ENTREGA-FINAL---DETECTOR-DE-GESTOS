"""Relatório das sessões do teste ao vivo na webcam (demo/teste_gestos.html, gravadas em resultados/webcam_ao_vivo/).

Duas medidas por gesto, com intervalo de confiança de 95% (Wilson), juntando todas as sessões e pessoas:
  - acerto quando pedido (sensibilidade): das vezes em que o gesto foi pedido, em quantas % ele disparou certo;
  - confiança medida (precisão): das vezes em que o sistema DISPAROU esse gesto (em qualquer momento: tentativas de
    outros gestos, gestos parecidos, preparo com a mão abaixada e o bloco sem gesto), em quantas % era mesmo ele.
A precisão é a resposta para "quando o sistema diz 2 dedos, ele acerta 90% das vezes?". Ela só vale como prova se a
sessão tiver também os gestos parecidos e o bloco sem gesto; sem eles, quase não há chance de disparo errado.

Veredito por gesto: "90% comprovado" quando o limite inferior do intervalo de 95% da precisão é >= 90%.

Uso:
    demo\\python.bat src\\relatorio_teste_ao_vivo.py
"""
import collections
import json
import math
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PASTA = RAIZ / "resultados" / "webcam_ao_vivo"


def wilson(certos, total, z=1.96):
    if total == 0:
        return None, None
    p = certos / total
    c = (p + z * z / (2 * total)) / (1 + z * z / total)
    m = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return round(100 * (c - m), 1), round(100 * (c + m), 1)


def main():
    sessoes = [json.loads(a.read_text(encoding="utf-8")) for a in sorted(PASTA.glob("*.json")) if a.name != "relatorio.json"]
    if not sessoes:
        print("Nenhuma sessão em", PASTA)
        return
    tentativas = [t for s in sessoes for t in s["tentativas"] if t.get("resultado") in ("acerto", "erro", "nao_disparou")]
    disparos = [d for s in sessoes for d in s["disparos"]]
    minutos_sem = sum((s.get("sem_gesto") or {}).get("segundos", 0) for s in sessoes) / 60
    falsos_sem = sum((s.get("sem_gesto") or {}).get("disparos", 0) for s in sessoes)
    # todo disparo é certo só se aconteceu no tempo de fazer o gesto pedido e era o gesto pedido
    parecidos = {g for s in sessoes for g in s["config"].get("parecidos", [])}

    def certo(d):
        return d["fase"] == "gesto" and d["alvo"] == d["gesto"] and d["alvo"] not in parecidos

    pedidos = sorted({t["pedido"] for t in tentativas if not t.get("parecido")})
    por_gesto = {}
    for g in pedidos:
        ts = [t for t in tentativas if t["pedido"] == g]
        ac = sum(t["resultado"] == "acerto" for t in ts)
        ds = [d for d in disparos if d["gesto"] == g]
        dc = sum(certo(d) for d in ds)
        lo, hi = wilson(dc, len(ds))
        por_gesto[g] = {
            "pedido_vezes": len(ts), "acerto_quando_pedido_pct": round(100 * ac / len(ts), 1) if ts else None,
            "acerto_ic95": wilson(ac, len(ts)),
            "disparou_vezes": len(ds), "precisao_pct": round(100 * dc / len(ds), 1) if ds else None, "precisao_ic95": [lo, hi],
            "confianca_media_mostrada_pct": round(100 * sum(d["confianca"] for d in ds) / len(ds), 1) if ds else None,
            "tempo_medio_s": round(sum(t["tempo_s"] for t in ts if t["resultado"] == "acerto") / max(ac, 1), 2) if ac else None,
            "errou_para": dict(collections.Counter(t["reconhecido"] for t in ts if t["resultado"] == "erro")),
            "disparos_errados_vieram_de": dict(collections.Counter(f'{d["fase"]}: {d["alvo"] or "sem gesto"}' for d in ds if not certo(d))),
            "veredito_90": None if lo is None else ("90% comprovado" if lo >= 90 else "90% NÃO comprovado"),
        }
    ps = [t for t in tentativas if t.get("parecido")]
    r = {
        "o_que_e": "Teste ao vivo na webcam com o modelo final (o do carro). Precisão = das vezes que o sistema disparou o "
                   "gesto, % em que era ele; intervalo de 95% de Wilson. '90% comprovado' = limite inferior >= 90%.",
        "sessoes": len(sessoes), "pessoas": sorted({s["pessoa"] for s in sessoes}),
        "tentativas": len(tentativas), "disparos": len(disparos),
        "acerto_geral_pct": round(100 * sum(t["resultado"] == "acerto" for t in tentativas if not t.get("parecido")) / max(1, sum(not t.get("parecido") for t in tentativas)), 1),
        "por_gesto": por_gesto,
        "gestos_parecidos": {"tentativas": len(ps), "dispararam_algo": sum(t["resultado"] == "erro" for t in ps),
                             "por_gesto": {g: dict(collections.Counter(t["reconhecido"] or "nada" for t in ps if t["pedido"] == g)) for g in sorted({t["pedido"] for t in ps})}},
        "sem_gesto": {"minutos": round(minutos_sem, 2), "disparos_falsos": falsos_sem,
                      "por_minuto": round(falsos_sem / minutos_sem, 2) if minutos_sem else None},
    }
    (PASTA / "relatorio.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
