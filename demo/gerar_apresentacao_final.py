"""Gera demo/apresentacao_final.html a partir de demo/apresentacao_final_modelo.html, com os números lidos de resultados/.

Reaproveita demo/gerar_apresentacao.py (sem alterá-lo) para montar os dados que o script dos slides antigos usa
(cenas 3D da CNN, matrizes etc.) e acrescenta os números da entrega final:
  - teste misto (resultados/pontos/misto_comparacao.json e misto_fusao_cnn_pre.json);
  - falsos disparos em vídeo contínuo (resultados/pontos/fluxo_combinado.json, videos_reais.json, fluxo_ipn_cnn_3d.json,
    resultados/dedos/fluxo_ipn.json);
  - comparação de arquiteturas (resultados/jester/comparacao_arquiteturas.json);
  - modelo usado ao vivo, com o veto da CNN sobre a contagem de dedos: acerto e confiança no teste misto
    (resultados/pontos/veto_outros.json) e falsos disparos em vídeo contínuo (resultados/pontos/fluxo_ao_vivo_veto2.json);
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
import graficos_finais as GF  # noqa: E402

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
    # portão da GRU (testado, não usado ao vivo): a GRU diz se é número; se for, a nota é dela; senão, a CNN sem os números
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
    linhas = []
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
        ("GRU + CNN, portão da GRU (mão parada)", f"{br(f['acc_mistura'])}%", "—", "—"),
        (f"Os {f.get('n_comandos')} gestos do carro · {br(f.get('car_n'), 0)} amostras do teste misto · falsos por minuto no IPN, só comandos do carro", None),
        ("CNN 3D pré-treinada + ajuste", f"{br(f.get('car_cnn'))}%", "—", f"{br(f.get('fl_cnn_carro'), 2)} / min"),
        ("GRU + CNN, números pela GRU (testada)", f"{br(f.get('car_gru'))}%", "—", "—"),
        ("CNN 3D + contagem de dedos + veto da CNN (testada)", f"{br(f.get('car_veto'))}%", "—", f"{br(f.get('fl_veto_carro'), 2)} / min"),
        ("Usada ao vivo: CNN 3D + contagem de dedos", f"{br(f.get('car_acc'))}%", "—", f"{br(f.get('fl_vivo_carro'), 2)} / min"),
    ]
    out = ['<table class="tabela tabModelos"><tr><th>Modelo</th><th>Acerto</th><th>Falsa ativação (Jester)</th><th>Falsos disparos (IPN, vídeo real)</th></tr>']
    for l in linhas:
        if l[1] is None:
            out.append(f'<tr class="bloco"><td colspan="4">{html.escape(l[0])}</td></tr>')
        else:
            cls = ' class="vencedora"' if l[0].startswith("Usada ao vivo") else ""
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
              ("GRU + CNN, números pela GRU", f["acc_pesos"], f["fl_final"]),
              ("Contagem de dedos + veto da CNN (testada)", f.get("acc_veto"), f.get("fl_veto")),
              ("Usada ao vivo: CNN 3D + contagem de dedos", f.get("acc_vivo"), f.get("fl_vivo"))]
    out = ['<table class="tabela"><tr><th>Modelo</th><th>Acerto</th><th>Falsos / min (IPN)</th></tr>']
    for nome, a, fl in linhas:
        vivo = nome.startswith("Usada ao vivo")
        cls = ' class="vencedora"' if vivo else ""
        falsos = br(fl, 2)
        acerto = "—" if a is None else f"{br(a)}%"
        out.append(f"<tr{cls}><td>{html.escape(nome)}</td><td>{acerto}</td><td>{falsos}</td></tr>")
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
    linhas = ['<tr class="vencedora"><td><b>OK</b> (polegar tocando o indicador)</td><td>geometria dos pontos</td>'
              '<td>ativa: bip, voz e 7 s para dar um comando</td></tr>']
    for n in com:
        quem = "contagem de dedos (pontos)" if n in NUMEROS else "CNN 3D"
        linhas.append(f'<tr><td><b>{html.escape(V.TRADUCAO.get(n, n))}</b></td><td>{quem}</td><td>{html.escape(acao.get(comandos[n], comandos[n]))}</td></tr>')
    sem = [V.TRADUCAO.get(n, n.replace("Volante: ", "Volante · ")) for n in nomes if n not in comandos and n not in G.SEM_GESTO]
    tabela = ('<table class="tabela tabGestos"><tr><th>Gesto</th><th>Quem reconhece</th><th>Comando no carro</th></tr>'
              + "".join(linhas) + "</table>")
    resto = (f'<p class="nota" style="font-size:16px;max-width:1260px"><b>{len(nomes)} classes no modelo.</b> Também reconhecidos, sem comando no carro: '
             + ", ".join(html.escape(s) for s in sem) + ". E duas classes de rejeição: sem gesto e outro movimento.</p>")
    contagem = {"n_numeros": sum(n in NUMEROS for n in nomes), "n_volante": sum(n.startswith("Volante") for n in nomes),
                "n_comandos": len(com)}
    contagem["n_jester"] = len(nomes) - contagem["n_numeros"] - contagem["n_volante"]
    return tabela, resto, len(nomes), contagem


def confianca_contagem():
    """Confiança da contagem de dedos usada ao vivo, com o veto da CNN, e a versão sem veto para comparar.

    Dados: resultados/pontos/veto_outros.json (src/medir_veto_outros.py), teste misto completo, sem rodar rede.
    """
    vo = (ler("resultados/pontos/veto_outros.json") or {}).get("variantes", {})
    # "com" = a versão usada ao vivo (sem veto); "sem" = a versão testada com o veto da CNN, desligada em 16/11
    com = vo.get("sem veto", {}).get("por_numero", {})
    sem = vo.get("veto B (ao vivo): CNN diz Outros gestos ou um movimento", {}).get("por_numero", {})

    def pct(v):
        return "—" if v is None else f"{br(v)}%"
    out = ['<table class="tabela"><tr><th>Número</th><th>Confiança<br>ao vivo</th><th>Mostrado e<br>passou, ao vivo</th>'
           '<th>Confiança<br>com veto (testada)</th><th>Mostrado e<br>passou, com veto</th></tr>']
    for k in ("1 dedo", "2 dedos", "3 dedos", "4 dedos"):
        c, s_ = com.get(k, {}), sem.get(k, {})
        nome = k + (" (sem comando)" if k == "3 dedos" else "")
        out.append(f"<tr><td>{html.escape(nome)}</td><td><b>{pct(c.get('confianca_pct'))}</b></td>"
                   f"<td>{pct(c.get('passou_quando_mostrado_pct'))}</td><td>{pct(s_.get('confianca_pct'))}</td>"
                   f"<td>{pct(s_.get('passou_quando_mostrado_pct'))}</td></tr>")
    out.append("</table>")
    return "".join(out), {}


def grafico_acerto(f):
    barras = [("CNN 3D|do zero", f["acc_cnn_zero"], "#c9d8f0"), ("GRU sobre|os pontos", f["acc_gru"], "#9db9ea"),
              ("CNN 3D|pré-treinada", f["acc_cnn_pre"], "#3d74d6"), ("GRU + CNN|(média)", f["acc_media"], "#3d74d6"),
              ("GRU + CNN|(números|pela GRU)", f["acc_pesos"], "#1e4fbf"), ("CNN 3D +|contagem|de dedos|usada ao vivo", f.get("acc_vivo"), "#0a1f44")]
    W, H, base, topo = 620, 490, 400, 40
    out = [f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Acerto por modelo no teste misto">']
    for v in (0, 25, 50, 75, 100):
        y = base - (base - topo) * v / 100
        out.append(f'<line x1="40" x2="{W - 10}" y1="{y}" y2="{y}" stroke="#e3e7ec"/><text x="32" y="{y + 5}" text-anchor="end" font-size="13" fill="#7b8594">{v}%</text>')
    largura = (W - 70) / len(barras)
    for i, (nome, v, cor) in enumerate(barras):
        if v is None:
            continue
        x, h = 50 + i * largura + 12, (base - topo) * v / 100
        out.append(f'<rect class="barra" x="{x}" y="{base - h}" width="{largura - 24}" height="{h}" fill="{cor}"/>')
        out.append(f'<text x="{x + (largura - 24) / 2}" y="{base - h - 10}" text-anchor="middle" font-size="20" font-weight="600" fill="#0a1f44">{br(v)}%</text>')
        for k, parte in enumerate(nome.split("|")):  # "|" separa as linhas do rótulo
            cor_t, peso = ("#1e4fbf", ' font-weight="700"') if parte == "usada ao vivo" else ("#364152", "")
            out.append(f'<text x="{x + (largura - 24) / 2}" y="{base + 24 + k * 19}" text-anchor="middle" font-size="14"{peso} fill="{cor_t}">{html.escape(parte)}</text>')
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
        out.append(f'<line x1="{x0}" x2="{x1}" y1="{py(a)}" y2="{py(a)}" stroke="#e3e7ec"/><text x="{x0 - 10}" y="{py(a) + 5}" text-anchor="end" font-size="13" fill="#7b8594">{a}%</text>')
    for w in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        out.append(f'<text x="{px(w)}" y="{y0 + 26}" text-anchor="middle" font-size="13" fill="#7b8594">{br(w)}</text>')
    out.append(f'<polyline points="{pts}" fill="none" stroke="#1e4fbf" stroke-width="3"/>')
    for w, a in sorted(curva.items()):
        out.append(f'<circle cx="{px(w)}" cy="{py(a)}" r="5" fill="#0a1f44"/>')
    melhor = max(curva, key=curva.get)
    out.append(f'<text x="{px(melhor)}" y="{py(curva[melhor]) - 16}" text-anchor="middle" font-size="17" font-weight="600" fill="#0a1f44">{br(curva[melhor])}%</text>')
    out.append(f'<text x="{(x0 + x1) / 2}" y="{H - 8}" text-anchor="middle" font-size="14" fill="#364152">peso da GRU (0 = só a CNN · 1 = só a GRU)</text></svg>')
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
    tab_gestos, resto_gestos, n_classes, contagem = tabela_gestos()
    reais = ler("resultados/pontos/videos_reais.json") or {}
    fi = ler("resultados/dedos/fluxo_ipn.json") or {}

    def fluxo(chave, campo):
        return flm.get(chave, {}).get(campo)
    final_chave = "Combinação (0.3 GRU; números 1.0 GRU)"
    vo = (ler("resultados/pontos/veto_outros.json") or {}).get("variantes", {})
    # 16/11 à noite: o veto saiu do modelo ao vivo (na webcam derrubava os números). "vivo" = sem veto; "veto" = testada
    vivo, veto = vo.get("sem veto", {}), vo.get("veto B (ao vivo): CNN diz Outros gestos ou um movimento", {})
    fvar = (ler("resultados/pontos/fluxo_ao_vivo_veto2.json") or {}).get("variantes", {})
    fv_todas = fvar.get("CNN 3D + contagem de dedos (confirmada em 1 análise)", {})
    fv, fvc = fv_todas.get("todos os gestos", {}), fv_todas.get("só comandos do carro", {})
    fs = fvar.get("CNN 3D + contagem de dedos + veto da CNN (janela mais longa)", {}).get("todos os gestos", {})
    f = {
        "n_teste": sum(v["n"] for v in misto["modelos"].get("GRU sobre pontos", {}).get("por_fonte", {}).values()),
        "n_classes": n_classes,
        "acc_gru": misto["modelos"].get("GRU sobre pontos", {}).get("todos"),
        "acc_cnn_zero": misto["modelos"].get("CNN 3D do zero", {}).get("todos"),
        "acc_cnn_pre": misto["modelos"].get("CNN 3D partindo do Jester inteiro", {}).get("todos"),
        "acc_media": fu["modelos"].get("Pontos + CNN (média)", {}).get("todos"),
        "acc_pesos": regras.get("pesos"), "acc_mistura": regras.get("mistura"), "acc_mistura_num": regras.get("mistura_numeros"),
        # configuração usada ao vivo (src/medir_veto_outros.py): CNN pré-treinada + contagem de dedos + veto da CNN
        # (gestos sem uso no carro somados em "Outros gestos"); "sem veto" = a versão anterior, testada
        "acc_vivo": vivo.get("acerto_36_classes_pct"), "acc_vivo_outros": vivo.get("acerto_pct"),
        "acc_vivo_num": vivo.get("acerto_numeros_pct"), "acc_vivo_resto": vivo.get("acerto_outros_pct"),
        "vivo_sem_numero": vivo.get("sem_numero_passou_numero_pct"),
        "acc_veto": veto.get("acerto_36_classes_pct"), "acc_veto_num": veto.get("acerto_numeros_pct"),
        "veto_sem_numero": veto.get("sem_numero_passou_numero_pct"),
        **{f"{p}{n}": d_.get("por_numero", {}).get(f"{n} dedo" + ("s" if n > 1 else ""), {}).get(c)
           for p, d_, c in (("cv", vivo, "confianca_pct"), ("pv", vivo, "passou_quando_mostrado_pct"),
                            ("cx", veto, "confianca_pct"), ("px", veto, "passou_quando_mostrado_pct"))
           for n in (1, 2, 3, 4)},
        # falsos por minuto em vídeo contínuo (src/fluxo_ao_vivo.py), mesmo critério das outras linhas (todos os gestos)
        "fl_vivo": fv.get("ipn_falsos_por_minuto"), "um_vivo": fv.get("ipn_um_dedo_trechos_certos"),
        "dois_vivo": fv.get("ipn_dois_dedos_trechos_certos"), "slovo_vivo": fv.get("slovo_falsos_por_minuto"),
        "fl_vivo_carro": fvc.get("ipn_falsos_por_minuto"),
        "intervalo_vivo": br(1 / fv["ipn_falsos_por_minuto"], 1) if fv.get("ipn_falsos_por_minuto") else None,
        "intervalo_vivo_carro": round(1 / fvc["ipn_falsos_por_minuto"]) if fvc.get("ipn_falsos_por_minuto") else None,
        "fl_veto": fs.get("ipn_falsos_por_minuto"), "um_veto": fs.get("ipn_um_dedo_trechos_certos"),
        "dois_veto": fs.get("ipn_dois_dedos_trechos_certos"),
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
    f.update(contagem)
    # modelo final: acerto nos gestos do carro, falsos por classe, arquiteturas e o teste ao vivo de 16/11
    import ao_vivo as V
    gc = ler("resultados/pontos/gestos_do_carro.json") or {}
    vivo_webcam = ler("resultados/webcam_ao_vivo/teste_16-11_sem_veto.json") or {}
    arq = ler("resultados/jester/comparacao_arquiteturas.json") or {}
    por_arq = {r["arquitetura"]: r for r in arq.get("resultados", [])}
    f.update(GF.campos(gc, vivo_webcam))
    graf_falsos, campos_falsos = GF.grafico_falsos(fvc, V.TRADUCAO)
    f.update(campos_falsos)
    tab_ok, tab_numeros, tab_movimentos = GF.gestos_principal(V.TRADUCAO)

    def pct_trechos(txt):
        a, b = (txt or "0/0").split("/")
        return round(100 * int(a) / int(b)) if int(b) else None
    f.update({
        "um_carro": fvc.get("ipn_um_dedo_trechos_certos"), "dois_carro": fvc.get("ipn_dois_dedos_trechos_certos"),
        "um_carro_pct": pct_trechos(fvc.get("ipn_um_dedo_trechos_certos")),
        "dois_carro_pct": pct_trechos(fvc.get("ipn_dois_dedos_trechos_certos")),
        "cnn_params_mi": GF.br(por_arq.get("cnn_3d", {}).get("parametros_mi"), 2),
        "arq_cnn3d": por_arq.get("cnn_3d", {}).get("top1_val"), "arq_larga": por_arq.get("cnn_3d_larga", {}).get("top1_val"),
        "arq_gru": por_arq.get("cnn_gru", {}).get("top1_val"), "arq_transformer": por_arq.get("cnn_transformer", {}).get("top1_val"),
        "arq_protocolo": arq.get("protocolo"), "jester_treino": d.get("dados", {}).get("videos_treino"),
    })
    # janela do OK (simulação), a régua dos gestos do carro nos outros modelos e os números do Jester
    jok = ler("resultados/pontos/fluxo_janela_ok.json") or {}
    gru_carro = (ler("resultados/pontos/gestos_do_carro_gru.json") or {}).get("modelos", {})
    jt, por_fonte = d.get("dados", {}), misto["modelos"].get("GRU sobre pontos", {}).get("por_fonte", {})
    total_1, total_2 = (fv.get("ipn_um_dedo_trechos_certos") or "/").split("/")[1], (fv.get("ipn_dois_dedos_trechos_certos") or "/").split("/")[1]
    f.update({
        "fl_ok_sem": jok.get("sem_janela", {}).get("falsos_por_minuto"), "fl_ok_com": jok.get("com_janela_do_ok", {}).get("falsos_por_minuto"),
        "fl_ok_abre": jok.get("ok_abriu_sozinho", {}).get("por_minuto"),
        "car_veto": gc.get("gestos_do_carro", {}).get("com_veto_pct"), "car_top5": gc.get("gestos_do_carro", {}).get("cnn_top5_pct"),
        "car_gru": gru_carro.get("GRU + CNN, números pela GRU", {}).get("gestos_do_carro_pct"),
        "car_gru_media": gru_carro.get("GRU + CNN (média 0,3 / 0,7)", {}).get("gestos_do_carro_pct"),
        "jester_total": jt.get("videos_total"), "jester_classes": jt.get("classes"), "jester_val": jt.get("videos_validacao"),
        "jester_teste": jt.get("videos_teste"), "jester_gb": jt.get("gigabytes"),
        "jester_frames_mi": round(jt.get("frames", 0) / 1e6, 1) or None,
        "mix_jester": por_fonte.get("Jester", {}).get("n"), "mix_ipn": por_fonte.get("IPN", {}).get("n"),
        "mix_hagrid": por_fonte.get("HaGRID", {}).get("n"), "mix_slovo": por_fonte.get("Slovo", {}).get("n"),
        "mix_driver": por_fonte.get("DriverMHG", {}).get("n"), "um_total": total_1, "dois_total": total_2,
    })
    # o que deu certo, na régua dos gestos do carro
    so_carro = "só comandos do carro"
    cnn_carro = fvar.get("CNN pré-treinada sozinha", {}).get(so_carro, {})
    f_sem, f_com = jok.get("sem_janela", {}).get("falsos"), jok.get("com_janela_do_ok", {}).get("falsos")
    f.update({
        "arq_n": len(arq.get("resultados", [])) or None, "car_gru_so": gru_carro.get("GRU sobre os pontos", {}).get("gestos_do_carro_pct"),
        "mov_falsos": cnn_carro.get("ipn_falsos"), "fl_cnn_carro": cnn_carro.get("ipn_falsos_por_minuto"),
        "fl_veto_carro": fvar.get("CNN 3D + contagem de dedos + veto da CNN (janela mais longa)", {}).get(so_carro, {}).get("ipn_falsos_por_minuto"),
        "ok_menos_pct": round(100 * (1 - f_com / f_sem)) if f_sem else None,
        "dados_mao": "Amostras do teste misto em que o MediaPipe achou a mão em pelo menos 70% dos quadros: "
                     + ", ".join(f"{k} {br(v)}%" for k, v in (misto.get("amostras_com_mao_por_fonte") or {}).items()) + ".",
        "car_num_ganho": GF.br(gc["numeros"]["acerto_pct"] - gc["numeros"]["cnn_sozinha_pct"]) if gc.get("numeros") else None,
    })
    tab_contagem, campos_contagem = confianca_contagem()
    f.update(campos_contagem)
    # um falso disparo a cada N minutos, na regra medida
    f["intervalo_final"] = round(1 / f["fl_final"]) if f.get("fl_final") else None
    trocas = {
        "<!--F:TABELA_MODELOS-->": tabela_modelos(d, f),
        "<!--F:TABELA_GESTOS-->": tab_gestos, "<!--F:RESTO_GESTOS-->": resto_gestos,
        "<!--F:TABELA_DADOS-->": tabela_dados(d),
        "<!--F:GRAFICO_GESTOS-->": GF.grafico_gestos(gc, V.TRADUCAO),
        "<!--F:GESTOS_NUMEROS-->": tab_numeros, "<!--F:GESTOS_MOVIMENTOS-->": tab_movimentos, "<!--F:GESTOS_OK-->": tab_ok,
        "<!--F:IC_CAMERA-->": GF.icone_camera(), "<!--F:IC_GESTO-->": GF.icone_mao("Swiping Right", 92), "<!--F:IC_OK-->": GF.icone_mao("OK", 92),
        "<!--F:IC_OK_GRANDE-->": GF.icone_mao("OK", 84),
        "<!--F:TABELA_ARQUITETURAS-->": GF.tabela_arquiteturas(arq),
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
    print(f"Gerado demo/apresentacao_final.html: ao vivo (sem veto) {f['acc_vivo']}% no teste misto, {f['fl_vivo']} falsos/min no IPN; "
          f"confiança da contagem {f['cv1']}/{f['cv2']}/{f['cv4']}% (1/2/4 dedos), {n_classes} classes")


if __name__ == "__main__":
    main()
