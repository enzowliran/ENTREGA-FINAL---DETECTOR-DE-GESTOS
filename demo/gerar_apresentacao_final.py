"""Gera demo/apresentacao_final.html a partir de demo/apresentacao_final_modelo.html, com os números lidos de resultados/.

Reaproveita demo/gerar_apresentacao.py (sem alterá-lo) para montar os dados que o script dos slides antigos usa
(cenas 3D da CNN, matrizes etc.) e acrescenta os números da entrega final:
  - teste misto (resultados/pontos/misto_comparacao.json e misto_fusao_cnn_pre.json);
  - falsos disparos em vídeo contínuo (resultados/pontos/fluxo_combinado.json, videos_reais.json, fluxo_ipn_cnn_3d.json,
    resultados/dedos/fluxo_ipn.json);
  - comparação de arquiteturas (resultados/jester/comparacao_arquiteturas.json);
  - gestos e comandos do carro (demo/player.py e as classes de resultados/pontos/modelo_final/cnn.pt).
Os números de fontes externas (NHTSA, NSC, Vi Bilägare, IAM RoadSmart, Euro NCAP) estão escritos no modelo, com a fonte.

Uso:
    python demo/gerar_apresentacao_final.py
"""
import html
import json
import pathlib
import sys
import types
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "demo"))
sys.path.insert(0, str(RAIZ / "src"))
import gerar_apresentacao as G  # noqa: E402

NUMEROS = ("One Finger", "Two Fingers", "Three Fingers", "Four Fingers")


def ler(caminho):
    p = RAIZ / caminho
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def br(x, casas=1):
    """Número no formato brasileiro; None vira travessão."""
    if x is None:
        return "—"
    return f"{x:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def dados_antigos():
    """Roda o gerador antigo só para pegar o dicionário de dados, sem gravar a apresentação antiga."""
    capturado = {}
    original = G.json.dumps

    def dumps(o, **kw):
        if isinstance(o, dict) and "grupo" in o:
            capturado["d"] = o
        return original(o, **kw)
    G.json = types.SimpleNamespace(**{k: getattr(json, k) for k in dir(json) if not k.startswith("__")})
    G.json.dumps = dumps
    gravar = pathlib.Path.write_text
    pathlib.Path.write_text = lambda self, *a, **k: None if self.name == "apresentacao.html" else gravar(self, *a, **k)
    try:
        G.main()
    finally:
        pathlib.Path.write_text = gravar
    return capturado["d"]


def regras_finais():
    """Acerto das regras de combinação no teste misto, das notas salvas por src/comparar_misto.py (fora do git)."""
    try:
        import numpy as np
        import pickle
        from comparar_misto import EXTRAS
        p = RAIZ / "data/pontos/misto_modelos"
        y, pg, pc = np.load(p / "teste_y.npy"), np.load(p / "gru_probs.npy"), np.load(p / "cnn_pre_probs.npy")
        nomes = list(pickle.load(open(RAIZ / "data/pontos/jester_amostra.pkl", "rb"))["nomes"]) + EXTRAS
    except Exception as e:  # sem os arquivos locais, os campos ficam com travessão
        print(f"Aviso: sem as notas do teste misto ({e}); as regras finais ficam sem número.")
        return {}
    num = np.array([nomes.index(n) for n in NUMEROS])
    # regra medida em vídeo contínuo: números com peso 1,0 da GRU, resto 0,3 da GRU, normalizada
    w = np.full(len(nomes), 0.3); w[num] = 1.0
    q = (1 - w) * pc + w * pg; q /= q.sum(1, keepdims=True)
    # regra ao vivo (demo/player.py): a GRU diz se é número; se for, a nota é dela; senão, a CNN sem os números
    pn = pg[:, num].sum(1, keepdims=True)
    co = pc.copy(); co[:, num] = 0; co /= np.maximum(co.sum(1, keepdims=True), 1e-9)
    m = (1 - pn) * co; m[:, num] = pg[:, num]
    e_num = np.isin(y, num)
    return {"pesos": round(100 * float((q.argmax(1) == y).mean()), 1),
            "mistura": round(100 * float((m.argmax(1) == y).mean()), 1),
            "mistura_numeros": round(100 * float((m.argmax(1) == y)[e_num].mean()), 1)}


def tabela_modelos(d, f):
    """Todos os modelos testados, por bloco, com acerto e falsa ativação quando existem."""
    arq = ler("resultados/jester/comparacao_arquiteturas.json") or {"resultados": []}
    fi = ler("resultados/dedos/fluxo_ipn.json") or {}
    cnn3d_ipn = (ler("resultados/pontos/fluxo_ipn_cnn_3d.json") or {}).get("resultados/jester/cnn_3d/melhor.pt", {})
    regras = ler("resultados/pontos/videos_reais.json") or {}
    hag = (ler("resultados/pontos/hagrid_amostra.json") or {}).get("teste_com_limiares_ajustados", {})
    jc = (ler("resultados/pontos/jester_coordenadas.json") or {}).get("coordenadas", {})
    linhas = [(f"Comparação de arquiteturas · Jester, treino curto ({br(d['pequeno_cfg']['videos'] / 1000, 0)} mil vídeos, "
               f"{d['pequeno_cfg']['epocas']} épocas) · falsa ativação = % dos vídeos sem gesto acima de 0,6", None)]
    for r in arq["resultados"]:
        nome = r["nome"] + (" (a escolhida)" if r["arquitetura"] == "cnn_3d" else "")
        linhas.append((nome, f"{br(r['top1_val'])}% (validação)", f"{br(r['falsa_ativacao_06'], 2)}% dos vídeos", "—"))
    linhas.append(("Modelos completos · falsos por minuto em fluxo contínuo: Jester simulado e IPN (13 pessoas, vídeo real)", None))
    mg, dd = fi.get("resultados/jester/cnn_3d_mais_gestos/melhor.pt", {}), fi.get("resultados/jester/cnn_3d_dedos/final.pt", {})
    linhas += [
        ("CNN 3D (Jester inteiro)", f"{br(d['teste']['top1'])}% (teste Jester)", f"{br(d['gatilho']['completo']['falsos'] / d['gatilho']['completo']['minutos'], 2)} / min",
         f"{br(cnn3d_ipn.get('falsos_por_minuto_sem_gesto'), 2)} / min"),
        ("CNN 3D + números (fotos HaGRID)", f"{br(d['novos']['top1_antigos'])}% (validação Jester)", f"{br(d['novos']['fpm'], 2)} / min",
         f"{br(mg.get('falsos_por_minuto_sem_gesto'), 2)} / min"),
        ("CNN 3D + números (vídeo real)", f"{br(d['dedos2']['jester_depois'])}% (validação Jester)",
         f"{br((ler('resultados/jester/cnn_3d_dedos/fluxo.json') or {}).get('novo', {}).get('falsos_por_min'), 2)} / min",
         f"{br(dd.get('falsos_por_minuto_sem_gesto'), 2)} / min"),
        ("Pontos da mão + regras (com punho)", f"{br(100 * hag.get('acerto_medio_numeros', 0))}% (números, fotos HaGRID)", "—",
         f"{br(regras.get('ipn', {}).get('falsos_por_minuto_sem_gesto'), 2)} / min"),
        ("GRU sobre os pontos (só Jester)", f"{br(jc.get('todos_top1'))}% (amostra Jester)", "—", "—"),
        (f"Teste misto · 5 bases, {f['n_classes']} classes, {br(f['n_teste'], 0)} amostras · falsos por minuto no IPN", None),
        ("GRU sobre os pontos", f"{br(f['acc_gru'])}%", "—", f"{br(f['fl_gru'], 2)} / min"),
        ("CNN 3D do zero", f"{br(f['acc_cnn_zero'])}%", "—", "—"),
        ("CNN 3D pré-treinada + ajuste", f"{br(f['acc_cnn_pre'])}%", "—", f"{br(f['fl_cnn_pre'], 2)} / min"),
        ("GRU + CNN (média 0,3 / 0,7)", f"{br(f['acc_media'])}%", "—", f"{br(f['fl_media'], 2)} / min*"),
        ("GRU + CNN, números pela GRU", f"{br(f['acc_pesos'])}%", "—", f"{br(f['fl_final'], 2)} / min"),
        ("Regra ao vivo: a GRU decide se é número", f"{br(f['acc_mistura'])}%", "—", "a medir"),
    ]
    out = ['<table class="tabela tabModelos"><tr><th>Modelo</th><th>Acerto</th><th>Falsa ativação (Jester)</th><th>Falsos disparos (IPN, vídeo real)</th></tr>']
    for l in linhas:
        if l[1] is None:
            out.append(f'<tr class="bloco"><td colspan="4">{html.escape(l[0])}</td></tr>')
        else:
            cls = ' class="vencedora"' if l[0].startswith("Regra ao vivo") or l[0].startswith("GRU + CNN, números") else ""
            out.append(f"<tr{cls}>" + "".join(f"<td>{html.escape(c)}</td>" for c in l) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


# Comandos do carro 3D na apresentação final (o mesmo mapa de MAPAS.carro3d no modelo da apresentação e de demo/carro3d.html).
MAPA_CARRO3D = {"Swiping Left": "anterior", "Swiping Right": "proxima", "Thumb Up": "volume_mais", "Thumb Down": "volume_menos",
                "Drumming Fingers": "tamborilar", "Stop Sign": "alternar", "Pushing Hand Away": "alternar", "Pulling Hand In": "ajuda",
                "One Finger": "pagina_1", "Two Fingers": "pagina_2", "Four Fingers": "pagina_4"}
ACAO_CARRO3D = {"anterior": "música anterior", "proxima": "próxima música",
                "volume_mais": "volume + (no ar, sobe o valor; mantido, atende a chamada)",
                "volume_menos": "volume − (no ar, desce o valor; mantido, recusa a chamada)",
                "tamborilar": "menu de músicas", "alternar": "toca/pausa, liga/desliga o ar, simula/encerra chamada, confirma no menu",
                "ajuda": "explica os comandos em voz alta", "pagina_1": "abre Música", "pagina_2": "abre Ar-condicionado", "pagina_4": "abre Telefone"}


def tabela_misto(f):
    """Só o bloco do teste misto (o slide de acerto); a tabela completa fica no apêndice."""
    linhas = [("GRU sobre os pontos", f["acc_gru"], f["fl_gru"]), ("CNN 3D do zero", f["acc_cnn_zero"], None),
              ("CNN 3D pré-treinada + ajuste", f["acc_cnn_pre"], f["fl_cnn_pre"]), ("GRU + CNN (média 0,3 / 0,7)", f["acc_media"], f["fl_media"]),
              ("GRU + CNN, números pela GRU", f["acc_pesos"], f["fl_final"]), ("Regra ao vivo: a GRU decide se é número", f["acc_mistura"], None)]
    out = ['<table class="tabela"><tr><th>Modelo</th><th>Acerto</th><th>Falsos / min (IPN)</th></tr>']
    for nome, a, fl in linhas:
        cls = ' class="vencedora"' if nome.startswith("Regra ao vivo") else ""
        falsos = "a medir" if nome.startswith("Regra ao vivo") else (f"{br(fl, 2)}" if fl is not None else "—")
        out.append(f"<tr{cls}><td>{html.escape(nome)}</td><td>{br(a)}%</td><td>{falsos}</td></tr>")
    out.append("</table>")
    return "".join(out)


def tabela_gestos():
    """Gestos que o modelo final reconhece e o comando de cada um no carro 3D."""
    import torch
    import ao_vivo as V
    nomes = list(torch.load(RAIZ / "resultados/pontos/modelo_final/cnn.pt", map_location="cpu", weights_only=False)["nomes"])
    comandos = MAPA_CARRO3D
    acao = ACAO_CARRO3D
    ordem = ["One Finger", "Two Fingers", "Four Fingers", "Swiping Right", "Swiping Left", "Thumb Up", "Thumb Down",
             "Stop Sign", "Pushing Hand Away", "Drumming Fingers", "Pulling Hand In"]
    com = [n for n in ordem if n in comandos and n in nomes]
    linhas = ['<tr class="vencedora"><td><b>OK 👌</b> (polegar tocando o indicador)</td><td>pontos da mão</td>'
              '<td>ativa: bip, voz e 7 s para dar um comando</td></tr>']
    for n in com:
        quem = "pontos + GRU" if n in NUMEROS else "CNN 3D"
        linhas.append(f'<tr><td><b>{html.escape(V.TRADUCAO.get(n, n))}</b></td><td>{quem}</td><td>{html.escape(acao.get(comandos[n], comandos[n]))}</td></tr>')
    sem = [V.TRADUCAO.get(n, n.replace("Volante: ", "Volante · ")) for n in nomes if n not in comandos and n not in G.SEM_GESTO]
    tabela = ('<table class="tabela tabGestos"><tr><th>Gesto</th><th>Quem reconhece</th><th>Comando no carro</th></tr>'
              + "".join(linhas) + "</table>")
    resto = (f'<p class="nota"><b>{len(nomes)} classes no modelo.</b> Também reconhecidos, sem comando no carro: '
             + ", ".join(html.escape(s) for s in sem) + ". E duas classes de rejeição: sem gesto e outro movimento.</p>")
    return tabela, resto, len(nomes)


def grafico_acerto(f):
    barras = [("CNN 3D do zero", f["acc_cnn_zero"], "#94a3b8"), ("GRU sobre os pontos", f["acc_gru"], "#60a5fa"),
              ("CNN 3D pré-treinada", f["acc_cnn_pre"], "#a78bfa"), ("GRU + CNN (média)", f["acc_media"], "#0b8a7f"),
              ("Regra ao vivo", f["acc_mistura"], "#db2777")]
    W, H, base, topo = 620, 470, 400, 40
    out = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Acerto por modelo no teste misto">']
    for v in (0, 25, 50, 75, 100):
        y = base - (base - topo) * v / 100
        out.append(f'<line x1="40" x2="{W - 10}" y1="{y}" y2="{y}" stroke="#dde4ea"/><text x="32" y="{y + 5}" text-anchor="end" font-size="14" fill="#93a2ad">{v}</text>')
    largura = (W - 70) / len(barras)
    for i, (nome, v, cor) in enumerate(barras):
        if v is None:
            continue
        x, h = 50 + i * largura + 12, (base - topo) * v / 100
        out.append(f'<rect class="barra" x="{x}" y="{base - h}" width="{largura - 24}" height="{h}" rx="8" fill="{cor}"/>')
        out.append(f'<text x="{x + (largura - 24) / 2}" y="{base - h - 10}" text-anchor="middle" font-size="22" font-weight="800" fill="#13232e">{br(v)}%</text>')
        palavras = nome.split(" ")
        meio = (len(palavras) + 1) // 2
        for k, parte in enumerate((" ".join(palavras[:meio]), " ".join(palavras[meio:]))):
            out.append(f'<text x="{x + (largura - 24) / 2}" y="{base + 24 + k * 19}" text-anchor="middle" font-size="15" fill="#5a6b77">{html.escape(parte)}</text>')
    out.append("</svg>")
    return "".join(out)


def curva_pesos(fu):
    curva = {float(k): v for k, v in (fu or {}).get("curva_pesos_todo_teste", {}).items()}
    if not curva:
        return "<p class='nota'>Sem a curva (resultados/pontos/misto_fusao_cnn_pre.json).</p>"
    W, H, x0, x1, y0, y1 = 760, 420, 70, 730, 360, 40
    lo, hi = 70.0, 95.0
    px = lambda w: x0 + (x1 - x0) * w
    py = lambda a: y0 - (y0 - y1) * (a - lo) / (hi - lo)
    pts = " ".join(f"{px(w):.1f},{py(a):.1f}" for w, a in sorted(curva.items()))
    out = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Acerto pelo peso da GRU">']
    for a in (70, 75, 80, 85, 90, 95):
        out.append(f'<line x1="{x0}" x2="{x1}" y1="{py(a)}" y2="{py(a)}" stroke="#dde4ea"/><text x="{x0 - 10}" y="{py(a) + 5}" text-anchor="end" font-size="14" fill="#93a2ad">{a}%</text>')
    for w in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        out.append(f'<text x="{px(w)}" y="{y0 + 26}" text-anchor="middle" font-size="14" fill="#93a2ad">{br(w)}</text>')
    out.append(f'<polyline points="{pts}" fill="none" stroke="#0b8a7f" stroke-width="4"/>')
    for w, a in sorted(curva.items()):
        out.append(f'<circle cx="{px(w)}" cy="{py(a)}" r="6" fill="#0b8a7f"/>')
    melhor = max(curva, key=curva.get)
    out.append(f'<text x="{px(melhor)}" y="{py(curva[melhor]) - 16}" text-anchor="middle" font-size="18" font-weight="800" fill="#13232e">{br(curva[melhor])}%</text>')
    out.append(f'<text x="{(x0 + x1) / 2}" y="{H - 8}" text-anchor="middle" font-size="15" fill="#5a6b77">peso da GRU (0 = só a CNN · 1 = só a GRU)</text></svg>')
    return "".join(out)


def tabela_dados(d):
    """As 5 bases: quantidades lidas dos arquivos e licenças."""
    b = d.get("bases", {})
    misto = ler("resultados/pontos/misto_comparacao.json") or {}
    jt = d["dados"]
    linhas = [
        ("Jester", f"{br(jt['videos_total'], 0)} vídeos, {jt['classes']} classes", "gestos com movimento (CNN)", "pesquisa; sem imagens nos slides"),
        ("IPN Hand", "50 pessoas, vídeo contínuo de webcam", "1 e 2 dedos, sem gesto, outros (pontos)", "CC BY 4.0"),
        ("HaGRID v2", f"{br(b['hagrid']['fotos_treino'] + b['hagrid']['fotos_validacao'], 0)} fotos baixadas de {br(b['hagrid']['imagens_no_arquivo'], 0)}", "1 a 4 dedos (pontos)", "atribuição; sem rostos"),
        ("Slovo", "vídeos de celular, \"sem sinal\"", "sem gesto (não disparar)", "CC BY-SA"),
        ("DriverMHG", "4 pessoas no disco (simulador de carro)", "micro-gestos do volante", "pesquisa; não redistribuir"),
    ]
    out = ['<table class="tabela tabDados"><tr><th>Base</th><th>Tamanho</th><th>Para que usamos</th><th>Licença</th></tr>']
    out += [f"<tr>{''.join(f'<td>{html.escape(c)}</td>' for c in l)}</tr>" for l in linhas]
    out.append("</table>")
    por = misto.get("amostras_com_mao_por_fonte", {})
    if por:
        out.append('<p class="nota">Amostras do teste misto em que o MediaPipe achou a mão em ≥ 70% dos quadros: '
                   + ", ".join(f"{k} {br(v)}%" for k, v in por.items()) + ".</p>")
    return "".join(out)


def main():
    d = dados_antigos()
    misto = ler("resultados/pontos/misto_comparacao.json") or {"modelos": {}}
    fu = ler("resultados/pontos/misto_fusao_cnn_pre.json") or {"modelos": {}}
    fl = (ler("resultados/pontos/fluxo_combinado.json") or {"modelos": {}})
    flm = fl["modelos"]
    regras = regras_finais()
    tab_gestos, resto_gestos, n_classes = tabela_gestos()
    reais = ler("resultados/pontos/videos_reais.json") or {}
    fi = ler("resultados/dedos/fluxo_ipn.json") or {}

    def fluxo(chave, campo):
        return flm.get(chave, {}).get(campo)
    final_chave = "Combinação (0.3 GRU; números 1.0 GRU)"
    f = {
        "n_teste": sum(v["n"] for v in misto["modelos"].get("GRU sobre pontos", {}).get("por_fonte", {}).values()),
        "n_classes": n_classes,
        "acc_gru": misto["modelos"].get("GRU sobre pontos", {}).get("todos"),
        "acc_cnn_zero": misto["modelos"].get("CNN 3D do zero", {}).get("todos"),
        "acc_cnn_pre": misto["modelos"].get("CNN 3D partindo do Jester inteiro", {}).get("todos"),
        "acc_media": fu["modelos"].get("Pontos + CNN (média)", {}).get("todos"),
        "acc_pesos": regras.get("pesos"), "acc_mistura": regras.get("mistura"), "acc_mistura_num": regras.get("mistura_numeros"),
        "fl_cnn_pre": fluxo("CNN pré-treinada", "ipn_falsos_por_minuto"), "fl_gru": fluxo("GRU sobre pontos", "ipn_falsos_por_minuto"),
        "fl_media": fluxo("Combinação (0.3 GRU)", "ipn_falsos_por_minuto"), "fl_final": fluxo(final_chave, "ipn_falsos_por_minuto"),
        "um_final": fluxo(final_chave, "ipn_um_dedo_trechos_certos"), "dois_final": fluxo(final_chave, "ipn_dois_dedos_trechos_certos"),
        "um_media": fluxo("Combinação (0.3 GRU)", "ipn_um_dedo_trechos_certos"), "dois_media": fluxo("Combinação (0.3 GRU)", "ipn_dois_dedos_trechos_certos"),
        "um_cnn": fluxo("CNN pré-treinada", "ipn_um_dedo_trechos_certos"), "dois_cnn": fluxo("CNN pré-treinada", "ipn_dois_dedos_trechos_certos"),
        "um_gru": fluxo("GRU sobre pontos", "ipn_um_dedo_trechos_certos"), "dois_gru": fluxo("GRU sobre pontos", "ipn_dois_dedos_trechos_certos"),
        "ipn_min": fl.get("ipn_minutos_sem_gesto"),
        "regras_fpm": reais.get("ipn", {}).get("falsos_por_minuto_sem_gesto"),
        "regras_um": reais.get("ipn", {}).get("apontar_One Finger", {}).get("pct_trechos_certos"),
        "regras_dois": reais.get("ipn", {}).get("apontar_Two Fingers", {}).get("pct_trechos_certos"),
        "d1_fpm": fi.get("resultados/jester/cnn_3d_mais_gestos/melhor.pt", {}).get("falsos_por_minuto_sem_gesto"),
        "d1_um": fi.get("resultados/jester/cnn_3d_mais_gestos/melhor.pt", {}).get("apontar_um_dedo", {}).get("disparos_certos"),
        "d2_fpm": fi.get("resultados/jester/cnn_3d_dedos/final.pt", {}).get("falsos_por_minuto_sem_gesto"),
        "d2_um": fi.get("resultados/jester/cnn_3d_dedos/final.pt", {}).get("apontar_um_dedo", {}).get("disparos_certos"),
        "teste_top1": d["teste"]["top1"],
        "driver_n": misto["modelos"].get("GRU sobre pontos", {}).get("por_fonte", {}).get("DriverMHG", {}).get("n"),
        "gru_params": (ler("resultados/pontos/jester_coordenadas.json") or {}).get("parametros"),
    }
    # um falso disparo a cada N minutos, na regra medida
    f["intervalo_final"] = round(1 / f["fl_final"]) if f.get("fl_final") else None
    trocas = {
        "<!--F:TABELA_MODELOS-->": tabela_modelos(d, f),
        "<!--F:TABELA_MISTO-->": tabela_misto(f),
        "<!--F:TABELA_GESTOS-->": tab_gestos, "<!--F:RESTO_GESTOS-->": resto_gestos,
        "<!--F:GRAFICO_ACERTO-->": grafico_acerto(f),
        "<!--F:CURVA_PESOS-->": curva_pesos(fu),
        "<!--F:TABELA_DADOS-->": tabela_dados(d),
    }
    modelo = (RAIZ / "demo/apresentacao_final_modelo.html").read_text(encoding="utf-8")
    for k, v in trocas.items():
        assert k in modelo, k
        modelo = modelo.replace(k, v)
    for k, v in f.items():
        token = "{{" + k + "}}"
        if isinstance(v, float):
            texto = br(v, 2) if k.startswith("fl_") or k.endswith("fpm") else br(v)
        elif isinstance(v, int):
            texto = br(v, 0)
        else:
            texto = "—" if v is None else str(v)
        modelo = modelo.replace(token, texto)
    sobrou = [t for t in set(__import__("re").findall(r"\{\{[a-z0-9_]+\}\}", modelo))]
    if sobrou:
        print("Aviso: campos sem valor:", sobrou)
    assert modelo.count("/*DADOS*/null") == 1
    (RAIZ / "demo/apresentacao_final.html").write_text(modelo.replace("/*DADOS*/null", json.dumps(d, ensure_ascii=False)), encoding="utf-8")
    print(f"Gerado demo/apresentacao_final.html: teste misto {f['acc_media']}% (média), regra ao vivo {f['acc_mistura']}%, "
          f"{f['fl_final']} falsos/min na regra medida, {n_classes} classes")


if __name__ == "__main__":
    main()
