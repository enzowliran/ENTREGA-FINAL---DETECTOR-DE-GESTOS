"""Gráficos e tabelas do modelo final para a apresentação (usados por demo/gerar_apresentacao_final.py).

Tudo aqui sai de arquivos de resultados/: nenhum número é escrito à mão.
  - resultados/pontos/gestos_do_carro.json (src/medir_gestos_do_carro.py): acerto nos 11 gestos do carro;
  - resultados/pontos/fluxo_ao_vivo_veto2.json (src/fluxo_ao_vivo.py): falsos disparos por classe, em vídeo contínuo;
  - resultados/jester/comparacao_arquiteturas.json (src/comparar_arquiteturas.py): arquiteturas treinadas do zero;
  - resultados/webcam_ao_vivo/teste_16-11_sem_veto.json: contagens do teste ao vivo de 16/11 à noite.
"""
import html

NUMEROS = ["One Finger", "Two Fingers", "Four Fingers"]
MOVIMENTOS = ["Swiping Right", "Swiping Left", "Thumb Up", "Thumb Down", "Stop Sign", "Pushing Hand Away",
              "Drumming Fingers", "Pulling Hand In"]
# o que cada gesto faz no carro, em poucas palavras (o mapa completo está em demo/player.py, COMANDOS_CARRO3D)
ACAO_CURTA = {"One Finger": "abre Música", "Two Fingers": "abre Ar-condicionado", "Four Fingers": "abre Telefone",
              "Swiping Right": "próxima música", "Swiping Left": "música anterior",
              "Thumb Up": "aumenta o volume; no ar, sobe a temperatura",
              "Thumb Down": "diminui o volume; no ar, baixa a temperatura",
              "Stop Sign": "toca ou pausa; liga ou desliga o ar", "Pushing Hand Away": "o mesmo comando do sinal de pare",
              "Drumming Fingers": "abre o menu de músicas", "Pulling Hand In": "explica os comandos em voz alta"}


def br(v, casas=1):
    return "—" if v is None else f"{v:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def grafico_gestos(gc, traducao):
    """Barras horizontais: acerto do modelo final em cada gesto do carro, em dois grupos (quem decide)."""
    pg = gc.get("por_gesto", {})
    W, x0, larg, alt, passo = 820, 270, 440, 26, 40
    grupos = [("Números · contagem de dedos", NUMEROS), ("Movimentos · CNN 3D", MOVIMENTOS)]
    y, out = 0, []
    for titulo, classes in grupos:
        y += 26
        out.append(f'<text x="0" y="{y}" font-size="13.5" font-weight="700" fill="#1e4fbf" letter-spacing="1.4">{html.escape(titulo.upper())}</text>')
        y += 14
        # do maior acerto para o menor, dentro de cada grupo
        classes = sorted(classes, key=lambda c: -(pg.get(c, {}).get("acerto_pct") or 0))
        for c in classes:
            v = pg.get(c, {}).get("acerto_pct")
            if v is None:
                continue
            cor = "#0a1f44" if c in NUMEROS else "#3d74d6"
            out.append(f'<text x="0" y="{y + 19}" font-size="18.5" fill="#364152">{html.escape(traducao.get(c, c))}</text>'
                       f'<rect x="{x0}" y="{y}" width="{larg}" height="{alt}" fill="#eef2f7"/>'
                       f'<rect class="barraH" x="{x0}" y="{y}" width="{larg * v / 100:.1f}" height="{alt}" fill="{cor}"/>'
                       f'<text x="{x0 + larg + 16}" y="{y + 20}" font-size="20" font-weight="600" fill="#0a1f44">{br(v, 0)}%</text>')
            y += passo
        y += 6
    out.append(f'<line x1="{x0}" x2="{x0}" y1="30" y2="{y - 12}" stroke="#d6dce4"/>')
    return (f'<svg viewBox="0 0 {W} {y}" width="{W}" height="{y}" role="img" aria-label="Acerto do modelo final em cada gesto do carro">'
            + "".join(out) + "</svg>")


# ---- pictograma da mão: palma cheia e dedos grossos, no azul dos slides ----
_AZUL, _MARINHO = "#1e4fbf", "#0a1f44"
_PALMA = "M72 184 L116 184 L131 144 L114 128 L76 128 L63 152 Z"
# cada dedo: (base, ponta esticada, ponta dobrada)
_DEDOS = {"polegar": ((66, 160), (26, 118), (52, 146)), "indicador": ((76, 128), (68, 60), (75, 112)),
          "medio": ((94, 126), (94, 48), (94, 110)), "anelar": ((112, 128), (120, 62), (113, 112)),
          "minimo": ((127, 140), (145, 90), (131, 126))}
_ORDEM = ["polegar", "indicador", "medio", "anelar", "minimo"]
_SETAS = {"direita": ("M44 26 H130", "130,13 152,26 130,39"), "esquerda": ("M130 26 H44", "44,13 22,26 44,39"),
          "frente": ("M168 150 V84", "155,84 168,62 181,84"), "tras": ("M168 74 V140", "155,140 168,162 181,140")}
# gesto -> (dedos esticados do polegar ao mínimo, opções do desenho)
_POSES = {"OK": ((1, 1, 1, 1, 1), {"ok": True}), "One Finger": ((0, 1, 0, 0, 0), {}), "Two Fingers": ((0, 1, 1, 0, 0), {}),
          "Four Fingers": ((0, 1, 1, 1, 1), {}), "Swiping Right": ((1, 1, 1, 1, 1), {"seta": "direita"}),
          "Swiping Left": ((1, 1, 1, 1, 1), {"seta": "esquerda"}), "Thumb Up": (None, {"polegar": "cima"}),
          "Thumb Down": (None, {"polegar": "baixo"}), "Stop Sign": ((1, 1, 1, 1, 1), {}),
          "Pushing Hand Away": ((1, 1, 1, 1, 1), {"seta": "frente"}), "Drumming Fingers": ((1, 1, 1, 1, 1), {"batidas": True}),
          "Pulling Hand In": ((1, 1, 1, 1, 1), {"seta": "tras"})}


def icone_mao(gesto, tam=56):
    """Pictograma pequeno da mão para um gesto (SVG)."""
    esticados, op = _POSES[gesto]
    if op.get("polegar"):   # punho fechado com o polegar para cima; para baixo, o mesmo desenho girado
        giro = ' transform="rotate(180 96 110)"' if op["polegar"] == "baixo" else ""
        corpo = (f'<g{giro}><rect x="64" y="96" width="84" height="78" rx="16" fill="{_AZUL}"/>'
                 f'<path d="M80 100 V40" stroke="{_AZUL}" stroke-width="24" stroke-linecap="round" fill="none"/>'
                 f'<path d="M100 116 H146 M100 135 H146 M100 154 H146" stroke="#fff" stroke-width="4" fill="none" opacity=".75"/></g>')
        return f'<svg viewBox="0 12 196 190" width="{tam}" height="{tam}" role="img" aria-hidden="true">{corpo}</svg>'
    tracos = []
    for nome, e in zip(_ORDEM, esticados):
        base, ponta, dobra = _DEDOS[nome]
        if op.get("ok") and nome == "polegar":
            tracos.append("M66 160 L44 136")
        elif op.get("ok") and nome == "indicador":
            tracos.append("M76 128 Q58 96 42 132")
        else:
            fim = ponta if e else dobra
            tracos.append(f"M{base[0]} {base[1]} L{fim[0]} {fim[1]}")
    extra = ""
    if op.get("seta"):
        linha, ponta = _SETAS[op["seta"]]
        extra += f'<path d="{linha}" fill="none" stroke="{_MARINHO}" stroke-width="9"/><polygon points="{ponta}" fill="{_MARINHO}"/>'
    if op.get("batidas"):
        for nome in _ORDEM[1:]:
            x, y = _DEDOS[nome][1]
            extra += f'<polyline points="{x - 8},{y - 30} {x},{y - 20} {x + 8},{y - 30}" fill="none" stroke="{_MARINHO}" stroke-width="6"/>'
    return (f'<svg viewBox="0 12 196 190" width="{tam}" height="{tam}" role="img" aria-hidden="true">'
            f'<path d="{_PALMA}" fill="{_AZUL}" stroke="{_AZUL}" stroke-width="14" stroke-linejoin="round"/>'
            f'<path d="{" ".join(tracos)}" fill="none" stroke="{_AZUL}" stroke-width="16" stroke-linecap="round"/>{extra}</svg>')


def icone_camera(tam=92):
    return (f'<svg viewBox="0 0 184 184" width="{tam}" height="{tam}" role="img" aria-hidden="true">'
            '<rect x="22" y="52" width="140" height="92" fill="#0a1f44"/><rect x="62" y="36" width="60" height="18" fill="#0a1f44"/>'
            '<circle cx="92" cy="98" r="30" fill="#1e4fbf" stroke="#9db9ea" stroke-width="5"/><circle cx="92" cy="98" r="12" fill="#0a1f44"/>'
            '<circle cx="142" cy="68" r="5" fill="#9db9ea"/></svg>')


def gestos_principal(traducao):
    """Slide Os gestos: três listas (ativação, números e movimentos), cada gesto com o desenho e o comando."""
    def tabela(linhas, cabecalho, classe=""):
        corpo = "".join(f'<tr{classe}><td class="ic">{icone_mao(g, 44)}</td><td><b>{html.escape(nome)}</b></td><td>{html.escape(acao)}</td></tr>'
                        for g, nome, acao in linhas)
        return f'<table class="tabela tabGestos tabPrincipal"><tr><th></th><th>Gesto</th><th>{cabecalho}</th></tr>{corpo}</table>'
    ok = tabela([("OK", "Sinal de OK", "ativa o sistema")], "O que faz", ' class="vencedora"')
    numeros = tabela([(c, traducao.get(c, c), ACAO_CURTA[c]) for c in NUMEROS], "Comando no carro")
    movimentos = tabela([(c, traducao.get(c, c), ACAO_CURTA[c]) for c in MOVIMENTOS], "Comando no carro")
    return ok, numeros, movimentos


def grafico_falsos(fluxo_todos, traducao):
    """Barras: de que classe são os falsos disparos em vídeo contínuo (IPN), no modelo final sem a janela do OK."""
    pc = dict(fluxo_todos.get("ipn_falsos_por_classe", {}))
    total = sum(pc.values())
    numeros = {"One Finger", "Two Fingers", "Three Fingers", "Four Fingers"}
    linhas = [(traducao.get(c, c) + (" (sem comando no carro)" if c == "Three Fingers" else ""), n, True)
              for c, n in sorted(pc.items(), key=lambda t: -t[1]) if c in numeros]
    resto = sum(n for c, n in pc.items() if c not in numeros)
    if resto:
        linhas.append(("Gestos de dois dedos em movimento", resto, False))
    W, x0, larg, passo = 700, 220, 380, 48
    maior = max((n for _, n, _ in linhas), default=1)
    out = []
    for i, (nome, n, e_num) in enumerate(linhas):
        y = i * passo
        out.append(f'<text x="0" y="{y + 22}" font-size="18" fill="#364152">{html.escape(nome)}</text>'
                   f'<rect class="barraH" x="{x0}" y="{y + 2}" width="{larg * n / maior:.1f}" height="28" fill="{"#0a1f44" if e_num else "#9db9ea"}"/>'
                   f'<text x="{x0 + larg * n / maior + 12:.1f}" y="{y + 23}" font-size="20" font-weight="600" fill="#0a1f44">{n}</text>')
    H = len(linhas) * passo
    out.append(f'<line x1="{x0}" x2="{x0}" y1="0" y2="{H - 8}" stroke="#d6dce4"/>')
    svg = f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Falsos disparos por classe">' + "".join(out) + "</svg>"
    de_numero = sum(n for _, n, e in linhas if e)
    if total and de_numero == total:
        frase = "Todos são números: uma mão parada à toa, com alguns dedos à mostra, parece um número. Nenhum movimento disparou sozinho."
    else:
        frase = f"{de_numero} dos {total} são números: uma mão parada à toa, com alguns dedos à mostra, parece um número."
    return svg, {"falsos_total": total, "falsos_numeros": de_numero, "falsos_frase": frase}


COMO_FUNCIONA = {
    "cnn_3d": "Quatro blocos de convoluções 3×3×3 (32, 64, 128 e 256 filtros): cada filtro olha altura, largura e tempo juntos",
    "cnn_3d_larga": "A mesma CNN 3D, com 50% mais filtros por bloco (48 a 384)",
    "cnn_3d_leve": "A mesma CNN 3D, com metade dos filtros por bloco (16 a 128)",
    "res_3d": "CNN 3D com atalhos: a entrada de cada bloco é somada à saída (blocos residuais)",
    "cnn_gru": "Uma CNN 2D resume cada quadro num vetor; uma GRU lê os 16 vetores em ordem",
    "cnn_transformer": "A mesma CNN 2D por quadro; no lugar da GRU, 2 camadas de atenção comparam todos os quadros entre si",
}


def tabela_arquiteturas(arq):
    """Apêndice: as arquiteturas treinadas do zero no mesmo protocolo (a comparação da entrega intermediária)."""
    linhas = []
    for r in arq.get("resultados", []):
        cls = ' class="vencedora"' if r["arquitetura"] == "cnn_3d" else ""
        nome = r["nome"] + (" (escolhida)" if r["arquitetura"] == "cnn_3d" else "")
        linhas.append(f'<tr{cls}><td>{html.escape(nome)}</td><td>{html.escape(COMO_FUNCIONA.get(r["arquitetura"], ""))}</td>'
                      f'<td>{br(r["parametros_mi"], 2)} mi</td><td>{br(r["top1_val"])}%</td><td>{br(r["falsa_ativacao_06"], 2)}%</td>'
                      f'<td>{br(r["latencia_1_ms"])} ms</td></tr>')
    return ('<table class="tabela tabArq"><tr><th>Arquitetura</th><th>Como funciona</th><th>Parâmetros</th><th>Acerto<br>(validação)</th>'
            '<th>Falsa<br>ativação</th><th>Tempo por<br>análise</th></tr>' + "".join(linhas) + "</table>")


def campos(gc, vivo):
    """Campos {{...}} do modelo final para o HTML."""
    g, n, m = gc.get("gestos_do_carro", {}), gc.get("numeros", {}), gc.get("movimentos", {})
    cnn = (vivo or {}).get("o_que_a_cnn_apontava_nessas_leituras", {})
    porg = (vivo or {}).get("comandos_por_gesto", {})
    return {
        "car_acc": g.get("acerto_pct"), "car_n": g.get("amostras"), "car_cmd": g.get("comando_certo_pct"),
        "car_cnn": g.get("cnn_sozinha_pct"),
        "car_num": n.get("acerto_pct"), "car_num_n": n.get("amostras"), "car_num_cnn": n.get("cnn_sozinha_pct"),
        "car_mov": m.get("acerto_pct"), "car_mov_n": m.get("amostras"), "car_mov_cnn": m.get("cnn_sozinha_pct"),
        "vivo_leituras": (vivo or {}).get("leituras_com_numero_pela_contagem"),
        "vivo_vetaria": (vivo or {}).get("leituras_em_que_o_veto_derrubaria_o_numero"),
        "vivo_outro_mov": cnn.get("Outro movimento"),
        "vivo_cmd_num": (vivo or {}).get("comandos_de_numero"), "vivo_cmd": (vivo or {}).get("comandos_disparados"),
        "vivo_min": round((vivo or {}).get("duracao_s", 0) / 60) or None,
        "vivo_um": porg.get("Um dedo"), "vivo_dois": porg.get("Dois dedos"), "vivo_quatro": porg.get("Quatro dedos"),
    }


def grafico_ganho(gc):
    """Barras: o que a contagem de dedos acrescentou à CNN 3D, nos números e nos 11 gestos do carro."""
    n, g = gc.get("numeros", {}), gc.get("gestos_do_carro", {})
    grupos = [("Números (1, 2 e 4 dedos)", n.get("cnn_sozinha_pct"), n.get("acerto_pct")),
              ("Os 11 gestos do carro", g.get("cnn_sozinha_pct"), g.get("acerto_pct"))]
    W, x0, larg = 1424, 470, 760
    out, y = [], 0
    for titulo, antes, depois in grupos:
        out.append(f'<text x="0" y="{y + 34}" font-size="19" font-weight="600" fill="#0a1f44">{html.escape(titulo)}</text>')
        for rotulo, v, cor, peso in (("CNN 3D sozinha", antes, "#9db9ea", "400"), ("Modelo final", depois, "#0a1f44", "600")):
            if v is None:
                continue
            out.append(f'<text x="{x0 - 16}" y="{y + 17}" text-anchor="end" font-size="15" fill="#7b8594">{rotulo}</text>'
                       f'<rect class="barraH" x="{x0}" y="{y}" width="{larg * v / 100:.1f}" height="24" fill="{cor}"/>'
                       f'<text x="{x0 + larg * v / 100 + 14:.1f}" y="{y + 19}" font-size="19" font-weight="{peso}" fill="#0a1f44">{br(v)}%</text>')
            y += 32
        y += 22
    y -= 22
    out.append(f'<line x1="{x0}" x2="{x0}" y1="0" y2="{y}" stroke="#d6dce4"/>')
    return f'<svg viewBox="0 0 {W} {y}" width="{W}" height="{y}" role="img" aria-label="Ganho da contagem de dedos">' + "".join(out) + "</svg>"
