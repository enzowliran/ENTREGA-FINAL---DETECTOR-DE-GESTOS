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
    W, x0, larg, alt, passo = 780, 250, 420, 20, 34
    grupos = [("Números · contagem de dedos", NUMEROS), ("Movimentos · CNN 3D", MOVIMENTOS)]
    y, out = 0, []
    for titulo, classes in grupos:
        y += 24
        out.append(f'<text x="0" y="{y}" font-size="12.5" font-weight="700" fill="#1e4fbf" letter-spacing="1.4">{html.escape(titulo.upper())}</text>')
        y += 12
        for c in classes:
            v = pg.get(c, {}).get("acerto_pct")
            if v is None:
                continue
            cor = "#0a1f44" if c in NUMEROS else "#3d74d6"
            out.append(f'<text x="0" y="{y + 15}" font-size="16" fill="#364152">{html.escape(traducao.get(c, c))}</text>'
                       f'<rect x="{x0}" y="{y}" width="{larg}" height="{alt}" fill="#eef2f7"/>'
                       f'<rect class="barraH" x="{x0}" y="{y}" width="{larg * v / 100:.1f}" height="{alt}" fill="{cor}"/>'
                       f'<text x="{x0 + larg + 14}" y="{y + 16}" font-size="17" font-weight="600" fill="#0a1f44">{br(v, 0)}%</text>')
            y += passo
        y += 6
    out.append(f'<line x1="{x0}" x2="{x0}" y1="30" y2="{y - 12}" stroke="#d6dce4"/>')
    return (f'<svg viewBox="0 0 {W} {y}" width="{W}" height="{y}" role="img" aria-label="Acerto do modelo final em cada gesto do carro">'
            + "".join(out) + "</svg>")


def gestos_principal(traducao):
    """Slide Os gestos: duas listas (números e movimentos), cada gesto com o seu comando."""
    def tabela(classes):
        linhas = "".join(f"<tr><td><b>{html.escape(traducao.get(c, c))}</b></td><td>{html.escape(ACAO_CURTA[c])}</td></tr>" for c in classes)
        return f'<table class="tabela tabGestos tabPrincipal"><tr><th>Gesto</th><th>Comando no carro</th></tr>{linhas}</table>'
    return tabela(NUMEROS), tabela(MOVIMENTOS)


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
