"""Gera demo/apresentacao.html a partir de demo/apresentacao_modelo.html, com os números lidos de resultados/.

Nenhum número da apresentação é digitado à mão: tudo vem dos arquivos de resultados. Os nomes do grupo
ficam em GRUPO, logo abaixo.

Uso:
    python demo/gerar_apresentacao.py
"""
import csv
import json
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
R = RAIZ / "resultados/jester"
GRUPO = ["Enzo Liran", "Isabela Giusti", "Maria Luiza Xavier"]  # em ordem alfabética

PT = {
    "Swiping Left": "Deslizar para a esquerda", "Swiping Right": "Deslizar para a direita", "Swiping Down": "Deslizar para baixo",
    "Swiping Up": "Deslizar para cima", "Pushing Hand Away": "Empurrar a mão", "Pulling Hand In": "Puxar a mão",
    "Sliding Two Fingers Left": "Dois dedos para a esquerda", "Sliding Two Fingers Right": "Dois dedos para a direita",
    "Sliding Two Fingers Down": "Dois dedos para baixo", "Sliding Two Fingers Up": "Dois dedos para cima",
    "Pushing Two Fingers Away": "Empurrar com dois dedos", "Pulling Two Fingers In": "Puxar com dois dedos",
    "Rolling Hand Forward": "Rolar a mão para frente", "Rolling Hand Backward": "Rolar a mão para trás",
    "Turning Hand Clockwise": "Girar a mão (horário)", "Turning Hand Counterclockwise": "Girar a mão (anti-horário)",
    "Zooming In With Full Hand": "Abrir a mão (zoom in)", "Zooming Out With Full Hand": "Fechar a mão (zoom out)",
    "Zooming In With Two Fingers": "Zoom in com dois dedos", "Zooming Out With Two Fingers": "Zoom out com dois dedos",
    "Thumb Up": "Polegar para cima", "Thumb Down": "Polegar para baixo", "Shaking Hand": "Acenar", "Stop Sign": "Sinal de pare",
    "Drumming Fingers": "Tamborilar os dedos", "No gesture": "Sem gesto", "Doing other things": "Outro movimento",
    "One Finger": "Um dedo", "Two Fingers": "Dois dedos", "Three Fingers": "Três dedos", "Four Fingers": "Quatro dedos", "Fist": "Punho",
}
SEM_GESTO = ("No gesture", "Doing other things")
DEMO = ["Swiping Left", "Swiping Right", "Stop Sign", "Thumb Up", "Thumb Down", "Drumming Fingers", "Pulling Hand In",
        "Pushing Hand Away", "Shaking Hand", "Zooming In With Full Hand", "Zooming Out With Full Hand"]
NOVOS = ["One Finger", "Two Fingers", "Three Fingers", "Four Fingers", "Fist"]
EQUIVALENTES = {"SwipeLeft": ("Swiping Left", "Sliding Two Fingers Left"), "SwipeRight": ("Swiping Right", "Sliding Two Fingers Right"),
                "FlickUp": ("Swiping Up", "Sliding Two Fingers Up"), "FlickDown": ("Swiping Down", "Sliding Two Fingers Down")}


def ler_json(caminho):
    return json.loads((RAIZ / caminho).read_text(encoding="utf-8"))


def historico(pasta):
    with (R / pasta / "historico.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def config(pasta):
    return ler_json(f"resultados/jester/{pasta}/config.json")


def fluxo_07_3():
    """Linha (limiar 0,7; 3 janelas) de cada modelo em comparacao_fluxo.txt: falsos/min e detecção."""
    saida, atual = {}, None
    for linha in (R / "comparacao_fluxo.txt").read_text(encoding="utf-8", errors="replace").splitlines():
        if linha.startswith("=== "):
            atual = linha[4:].strip()
        elif atual and re.match(r"\s*0\.70\s+3\s", linha) and atual not in saida:
            campos = linha.split()
            saida[atual] = {"fpm": float(campos[2]), "falsos": int(campos[3]), "det": float(campos[4])}
    return saida


def matriz(caminho, linhas_classes, colunas_classes):
    """Agrupa a matriz completa: 'Outros gestos' junta as classes que não aparecem; 'Sem gesto' junta as duas negativas."""
    m = ler_json(caminho)
    nomes, c = m["nomes"], m["contagens"]
    def grupo(nome, lista):
        if nome in SEM_GESTO:
            return "Sem gesto"
        return PT[nome] if nome in lista else "Outros gestos"
    rotulos_l = [PT[n] for n in linhas_classes] + ["Sem gesto"]
    rotulos_c = [PT[n] for n in colunas_classes] + ["Outros gestos", "Sem gesto"]
    tabela = {rl: {rc: 0 for rc in rotulos_c} for rl in rotulos_l}
    for i, ni in enumerate(nomes):
        gl = grupo(ni, linhas_classes)
        if gl not in tabela:
            continue
        for j, nj in enumerate(nomes):
            tabela[gl][grupo(nj, colunas_classes)] += c[i][j]
    linhas = []
    for rl in rotulos_l:
        total = sum(tabela[rl].values())
        linhas.append([round(100 * tabela[rl][rc] / total, 1) if total else 0 for rc in rotulos_c])
    return {"linhas_rotulos": rotulos_l, "colunas_rotulos": rotulos_c, "valores": linhas, "n": m["n"], "acerto": round(100 * m["acerto"], 1)}


def main():
    d = {"grupo": GRUPO, "dados": ler_json("resultados/jester/dados_jester.json")}

    # Passo 1: primeiros testes, mesmo subconjunto (20 mil vídeos de treino, 5 mil de validação, 8 épocas).
    d["pequeno"] = [{"nome": "CNN 3D", "top1": round(100 * float(historico("cnn_3d_pequeno")[-1]["top1_val"]), 1)},
                    {"nome": "CNN 2D + Transformer", "top1": round(100 * float(historico("cnn_transformer_pequeno")[-1]["top1_val"]), 1)},
                    {"nome": "CNN 2D + GRU", "top1": None}]
    d["pequeno_cfg"] = {"videos": config("cnn_3d_pequeno")["limite_treino"], "epocas": config("cnn_3d_pequeno")["epocas"]}
    # Comparação justa de arquiteturas (src/comparar_arquiteturas.py); a GRU com taxa 1e-3 travou e foi refeita com 3e-4.
    d["arquiteturas"] = ler_json("resultados/jester/comparacao_arquiteturas.json")
    # Ativações reais do modelo para um vídeo sintético (src/ativacoes_exemplo.py), para os slides da CNN 3D.
    d["ativacoes"] = ler_json("resultados/jester/cnn_3d/ativacoes_exemplo.json")
    for g in d["ativacoes"]["probabilidades"]:
        g["gesto_pt"] = PT[g["gesto"]]
    d["arquiteturas"]["gru_lr1e3"] = round(100 * max(float(l["top1_val"]) for l in historico("arq_cnn_gru_lr1e-3")), 1)

    # Passo 2: custo extra para disparos sem gesto, no mesmo subconjunto.
    pen = []
    for pasta, rotulo in (("cnn_3d_pequeno", "Sem custo"), ("cnn_3d_pequeno_p1", "Custo 1"), ("cnn_3d_pequeno_p2", "Custo 2"), ("cnn_3d_pequeno_punido", "Custo 5")):
        a = ler_json(f"resultados/jester/{pasta}/avaliacao_falsas.json")
        fa = next(x for x in a["curva"] if x["limiar"] == 0.6)
        pen.append({"rotulo": rotulo, "peso": config(pasta).get("peso_falso", 0), "top1": round(100 * a["top1"], 1), "fa": round(100 * fa["falsa_ativacao"], 2), "n": a["n"]})
    d["penalidade"] = pen

    # Passo 3: mais dados e mais épocas, com o gatilho de 0,7 e 3 janelas, nos mesmos 1.500 vídeos com gesto.
    f = fluxo_07_3()
    calib = list(csv.DictReader((R / "cnn_3d/calibracao_fluxo.csv").open(encoding="utf-8")))
    grande = next(l for l in calib if float(l["limiar"]) == 0.7 and int(l["janelas_seguidas"]) == 3)
    escala = []
    for pasta, rotulo in (("cnn_3d_pequeno", "Pequeno"), ("cnn_3d_medio", "Médio")):
        cfg = config(pasta)
        escala.append({"rotulo": rotulo, "videos": cfg["limite_treino"], "epocas": cfg["epocas"], "fpm": f[pasta]["fpm"], "det": round(100 * f[pasta]["det"], 1)})
    escala.append({"rotulo": "Grande", "videos": d["dados"]["videos_treino"], "epocas": config("cnn_3d")["epocas"],
                   "fpm": round(float(grande["falsos_por_min"]), 2), "det": round(100 * float(grande["gestos_detectados"]), 1)})
    d["escala"] = escala

    # Passo 4: o treino grande.
    h = historico("cnn_3d")
    d["curva"] = {"treino": [round(100 * float(l["top1_treino"]), 1) for l in h], "val": [round(100 * float(l["top1_val"]), 1) for l in h],
                  "minutos_epoca": round(sum(float(l["segundos"]) for l in h) / len(h) / 60, 1)}
    t = ler_json("resultados/jester/cnn_3d/teste.json")
    d["teste"] = {"top1": round(100 * t["top1"], 1), "top5": round(100 * t["top5"], 1), "n": t["n"]}

    # Passo 5: o gatilho, com as 30 combinações medidas e o resultado no fluxo completo.
    d["gatilho"] = {"pontos": [{"limiar": float(l["limiar"]), "janelas": int(l["janelas_seguidas"]), "fpm": round(float(l["falsos_por_min"]), 3),
                                "det": round(100 * float(l["gestos_detectados"]), 1)} for l in calib]}
    fl = ler_json("resultados/jester/cnn_3d_mais_gestos/fluxo.json")["origem"]
    d["gatilho"]["completo"] = {"falsos": fl["falsos"], "minutos": round(fl["minutos"], 1), "det": round(100 * fl["deteccao_antigos"], 1),
                                "errada": round(100 * fl["classe_errada"], 1), "videos": fl["videos"]}
    roc = ler_json("resultados/jester/cnn_3d/roc_auc.json")["teste"]
    d["roc"] = {"auc": round(roc["auc_gesto_x_sem_gesto"], 4)}

    d["matriz"] = matriz("resultados/jester/cnn_3d/matriz_confusao_validacao.json", DEMO, DEMO)
    demo_pt = {PT[n] for n in DEMO} | {"Zoom in com a mão", "Zoom out com a mão"}  # nomes usados em ranking_gestos.csv
    with (R / "cnn_3d/ranking_gestos.csv").open(encoding="utf-8") as fr:
        d["ranking"] = [{"gesto": l["gesto"], "det": round(100 * float(l["detectados"]), 1)} for l in csv.DictReader(fr) if l["gesto"] in demo_pt]
    d["ranking"].sort(key=lambda x: -x["det"])

    # O que não funcionou.
    lpg = ler_json("resultados/jester/cnn_3d/limiar_por_gesto.json")
    ed = ler_json("resultados/jester/cnn_3d_esq_dir/resumo.json")
    melhor_ed = max(ed["historico"], key=lambda x: (x["esquerda"] + x["direita"]) / 2)
    certos, total = 0, 0
    with (RAIZ / "resultados/drivermhg/jester_sem_ajuste.csv").open(encoding="utf-8") as fr:
        for l in csv.DictReader(fr):
            if l["fonte"] == "rgb" and l["modo"] == "normal" and l["gesto"] in EQUIVALENTES:
                total += 1
                certos += l["predito"] in EQUIVALENTES[l["gesto"]]
    d["descartados"] = {
        "limiar": {"unico": {"falsos": lpg["unico"]["falsos"], "det": round(100 * lpg["unico"]["deteccao"], 1)},
                   "por_gesto": {"falsos": lpg["por_gesto_fora_da_amostra"]["falsos"], "det": round(100 * lpg["por_gesto_fora_da_amostra"]["deteccao"], 1)}},
        "esqdir": {"origem": round(100 * ed["origem_validacao"]["top1"], 1), "ajustado": round(100 * melhor_ed["top1"], 1)},
        "driver": {"certos": certos, "total": total}}

    # Passo 6: gestos novos.
    r1 = ler_json("resultados/jester/cnn_3d_mais_gestos_teste/resumo.json")
    r2 = ler_json("resultados/jester/cnn_3d_mais_gestos_teste2/resumo.json")
    rf = ler_json("resultados/jester/cnn_3d_mais_gestos/resumo.json")
    salvo = __import__("torch").load(R / "cnn_3d_mais_gestos/melhor.pt", map_location="cpu", weights_only=False)["epoca"]
    fn = ler_json("resultados/jester/cnn_3d_mais_gestos/fluxo.json")
    d["novos"] = {"teste1": round(100 * r1[-1]["novas_acerto"], 1), "teste2": round(100 * r2[-1]["novas_acerto"], 1),
                  "epocas": [{"novos": round(100 * x["novas_acerto"], 1), "antigos": round(100 * x["jester_top1"], 1)} for x in rf],
                  "escolhida": salvo, "fotos_treino": None,
                  "falsos": fn["novo"]["falsos"], "fpm": round(fn["novo"]["falsos_por_min"], 2), "falsos_de_novos": fn["novo"]["falsos_de_gestos_novos"],
                  "top1_antigos": round(100 * fn["top1_antigos_validacao"], 1),
                  "det07": {PT[k]: round(100 * v, 1) for k, v in rf[salvo]["novas_detectadas_07"].items()}}
    d["novos"]["fotos_treino"] = sum(1 for _ in (RAIZ / "data/HaGRID/train").rglob("*.jpg")) if (RAIZ / "data/HaGRID/train").exists() else None
    # As três bases (src/estatisticas_bases.py) e de que classe do HaGRID veio cada saída nova.
    d["bases"] = ler_json("resultados/estatisticas_bases.json")
    import sys
    sys.path.insert(0, str(RAIZ / "src"))
    from treinar_gestos_novos import MAPA
    usadas = d["bases"]["hagrid"]["usadas"]
    destinos = NOVOS + ["Thumb Up", "Thumb Down", "Stop Sign", "No gesture", "Doing other things"]
    d["mapa_hagrid"] = [{"destino": PT[dest], "tipo": "novo" if dest in NOVOS else "negativo" if dest in SEM_GESTO else "antigo",
                         "origens": [{"classe": o, "fotos": usadas.get(o, {}).get("train", 0)} for o, x in MAPA.items() if x == dest]}
                        for dest in destinos]
    d["treino_novos"] = {"jester_por_epoca": config("cnn_3d_mais_gestos")["jester_por_epoca"], "epocas": config("cnn_3d_mais_gestos")["epocas"],
                         "lr": config("cnn_3d_mais_gestos")["lr"], "lr_cabeca": config("cnn_3d_mais_gestos")["lr_cabeca"],
                         "saidas_antigas": len(salvo_nomes := __import__("torch").load(R / "cnn_3d/melhor.pt", map_location="cpu", weights_only=False)["nomes"]),
                         "saidas": len(salvo_nomes) + len(NOVOS)}
    # 2ª tentativa dos dedos (src/treinar_dedos.py), com vídeo real; e as sessões do testador na webcam, por modelo.
    origem2 = ler_json("resultados/jester/cnn_3d_dedos/origem.json")
    final2 = ler_json("resultados/jester/cnn_3d_dedos/resumo.json")[-1]
    fluxo2 = ler_json("resultados/jester/cnn_3d_dedos/fluxo.json")
    fluxo_ipn = ler_json("resultados/dedos/fluxo_ipn.json")
    sessoes = {}
    for pasta in sorted((RAIZ / "resultados/webcam").glob("*_*")):
        if not (pasta / "resumo.json").exists():
            continue
        modelo = ler_json(pasta.relative_to(RAIZ).as_posix() + "/resumo.json")["modelo"].replace("\\", "/")
        with (pasta / "tentativas.csv").open(encoding="utf-8-sig") as f:
            sessoes.setdefault(modelo, []).extend(l for l in csv.DictReader(f) if l["gesto"] in ("One Finger", "Two Fingers"))
    def webcam(modelo):
        t = sessoes.get(modelo, [])
        com_posicao = [l for l in t if l.get("melhor_posicao", "").strip()]
        return {"tentativas": len(t), "acertos": sum(l["resultado"] == "acerto" for l in t),
                "primeiro": sum(l["melhor_posicao"] == "1" for l in com_posicao), "com_posicao": len(com_posicao)}
    antigo, novo = "resultados/jester/cnn_3d_mais_gestos/melhor.pt", "resultados/jester/cnn_3d_dedos/final.pt"
    d["dedos2"] = {
        "ipn": {k: [round(100 * origem2[f"{p}_{g}_top1"], 1), round(100 * final2[f"{p}_{g}_top1"], 1)]
                for k, (p, g) in {"ipn_um": ("ipn", "One Finger"), "ipn_dois": ("ipn", "Two Fingers"),
                                  "sinais_um": ("sinais", "One Finger"), "sinais_dois": ("sinais", "Two Fingers")}.items()},
        "jester_antes": d["novos"]["top1_antigos"], "jester_depois": round(100 * fluxo2["top1_antigos_validacao"], 1),
        "falsos_real_antes": fluxo_ipn[antigo]["falsos_por_minuto_sem_gesto"], "falsos_real_depois": fluxo_ipn[novo]["falsos_por_minuto_sem_gesto"],
        "minutos_real": fluxo_ipn[antigo]["minutos_sem_gesto"], "falsos_jester": round(fluxo2["origem"]["falsos_por_min"], 2),
        "webcam_antes": webcam(antigo), "webcam_depois": webcam(novo),
    }
    d["jester_classes"] =[{"gesto": PT[g], "n": n} for g, n in d["bases"]["jester"]["treino_por_classe"].items()]
    d["matriz2"] = matriz("resultados/jester/cnn_3d_mais_gestos/matriz_confusao_hagrid_val.json", NOVOS + ["Stop Sign", "Thumb Up", "Thumb Down"], NOVOS + ["Stop Sign", "Thumb Up", "Thumb Down"])

    modelo = (RAIZ / "demo/apresentacao_modelo.html").read_text(encoding="utf-8")
    assert modelo.count("/*DADOS*/null") == 1
    (RAIZ / "demo/apresentacao.html").write_text(modelo.replace("/*DADOS*/null", json.dumps(d, ensure_ascii=False)), encoding="utf-8")
    print(f"Gerado demo/apresentacao.html: teste {d['teste']['top1']}%, {d['gatilho']['completo']['falsos']} falsos em {d['gatilho']['completo']['minutos']} min, "
          f"gestos novos {d['novos']['top1_antigos']}% nos antigos")


if __name__ == "__main__":
    main()
