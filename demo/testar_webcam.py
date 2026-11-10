"""Testador guiado na webcam: pede um gesto de cada vez, roda o modelo ao vivo e anota o resultado.

Usa a mesma regra de disparo das demos (confiança >= 0,7 em 3 janelas seguidas, depois uma pausa).
Cada tentativa tem um tempo de preparo, com a mão abaixada, e um tempo para fazer o gesto:
  - acerto: o primeiro disparo no tempo do gesto foi a classe pedida (anota quanto tempo levou);
  - erro: o primeiro disparo foi outra classe (anota qual);
  - não disparou: nada passou do gatilho.
No fim vem um bloco "sem gesto", em que a pessoa age normalmente: todo disparo ali é falso.

Não grava vídeo nem imagem (a câmera mostra o rosto). Grava só números, em resultados/webcam/<pessoa>_<data>/:
  tentativas.csv, disparos.csv e resumo.json. Com --resumo, junta as sessões de todos os integrantes em
  resultados/webcam/resumo_geral.json e todas_tentativas.csv (refeitos a cada vez, fora do git).

Teclas: Espaço pula a tentativa, R recomeça a tentativa atual, P pausa, Q ou Esc encerra e salva o que já foi feito.

Uso:
    python demo/testar_webcam.py --pessoa enzo
    python demo/testar_webcam.py --pessoa enzo --conjunto dedos --tentativas 5
    python demo/testar_webcam.py --resumo        (junta todas as sessões já feitas)
"""
import argparse
import collections
import csv
import datetime
import json
import random
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ao_vivo as V

RAIZ = Path(__file__).resolve().parent.parent
PASTA = RAIZ / "resultados/webcam"
DICAS = {
    "One Finger": "Indicador levantado, palma para a câmera",
    "Two Fingers": "Indicador e médio levantados",
    "Three Fingers": "Indicador, médio e anelar levantados",
    "Four Fingers": "Quatro dedos levantados, polegar dobrado",
    "Fist": "Mão fechada, parada",
    "Stop Sign": "Mão aberta, palma para a câmera, parada",
    "Thumb Up": "Joinha, parado",
    "Thumb Down": "Polegar para baixo, parado",
    "Swiping Left": "Passe a mão da direita para a esquerda",
    "Swiping Right": "Passe a mão da esquerda para a direita",
    "Pulling Hand In": "Mão aberta, puxe em direção ao corpo",
    "Pushing Hand Away": "Mão aberta, empurre para longe de você",
    "Drumming Fingers": "Tamborile os dedos no ar",
    "Shaking Hand": "Acene",
    "Zooming In With Full Hand": "Mão fechada que abre",
    "Zooming Out With Full Hand": "Mão aberta que fecha",
}
CONJUNTOS = {
    "dedos": ["One Finger", "Two Fingers", "Three Fingers", "Four Fingers", "Fist"],
    "demos": ["One Finger", "Two Fingers", "Stop Sign", "Thumb Up", "Thumb Down", "Swiping Left", "Swiping Right",
              "Pulling Hand In", "Pushing Hand Away", "Drumming Fingers"],
}
CONJUNTOS["todos"] = list(dict.fromkeys(CONJUNTOS["dedos"] + CONJUNTOS["demos"] + ["Shaking Hand", "Zooming In With Full Hand", "Zooming Out With Full Hand"]))
VERDE, VERMELHO, CINZA, AMARELO, BRANCO = (90, 220, 130), (255, 95, 95), (190, 190, 190), (255, 205, 90), (240, 240, 235)


class Gatilho:
    """O mesmo de demo/player.py, sem repetição: N janelas seguidas da mesma classe e uma pausa depois."""

    def __init__(self, limiar, seguidas, pausa):
        self.limiar, self.seguidas, self.pausa = limiar, seguidas, pausa
        self.atual, self.contagem, self.fim_pausa, self.armado = None, 0, 0.0, True

    def atualizar(self, nome, confianca, agora):
        if nome in V.SEM_GESTO or confianca < self.limiar:
            self.atual, self.contagem, self.armado = None, 0, True
            return None
        if nome != self.atual:
            self.atual, self.contagem, self.armado = nome, 0, True
        self.contagem += 1
        if self.armado and self.contagem >= self.seguidas and agora >= self.fim_pausa:
            self.armado, self.fim_pausa = False, agora + self.pausa
            return nome
        return None


class Modelo:
    """Buffer de quadros a 12 por segundo e inferência em várias janelas, como nas demos."""

    def __init__(self, args):
        self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        self.modelo, self.nomes, info = V.carregar_modelo(Path(args.modelo), self.dispositivo)
        self.janelas = args.janelas or ([1.0, 1.5, 3.0] if self.dispositivo == "cuda" else [1.0, 1.5])
        self.minimo = int(round(min(self.janelas) * V.QUADROS_POR_SEGUNDO))
        self.buffer = collections.deque(maxlen=int(round(max(self.janelas) * V.QUADROS_POR_SEGUNDO)))
        self.historico = collections.deque(maxlen=args.suavizar)
        self.indices_gesto = [i for i, n in enumerate(self.nomes) if n not in V.SEM_GESTO]
        self.ultimo_quadro, self.ultima, self.intervalo = 0.0, 0.0, args.intervalo
        self.media = None
        print(f"Modelo: {Path(args.modelo).parent.name} (validação {info['top1_val']:.1%}), {self.dispositivo}", flush=True)

    def receber(self, quadro, agora):
        """Devolve (classe, confiança, probabilidades) quando há uma inferência nova; senão None."""
        if agora - self.ultimo_quadro >= 1 / V.QUADROS_POR_SEGUNDO:
            self.buffer.append(V.preparar_quadro(quadro))
            self.ultimo_quadro = agora
        if len(self.buffer) < self.minimo or agora - self.ultima < self.intervalo:
            return None
        self.ultima = agora
        lista, lotes = list(self.buffer), []
        for duracao in self.janelas:
            n = int(round(duracao * V.QUADROS_POR_SEGUNDO))
            if len(lista) >= n:
                indices = np.round(np.linspace(len(lista) - n, len(lista) - 1, V.N_FRAMES)).astype(int)
                lotes.append(np.stack([lista[i] for i in indices]))
        with torch.no_grad(), torch.autocast(self.dispositivo, dtype=torch.bfloat16, enabled=self.dispositivo == "cuda"):
            x = (torch.from_numpy(np.stack(lotes)).to(self.dispositivo) - 127.5) / 127.5
            todas = self.modelo(x).float().softmax(1).cpu().numpy()
        escolhida = int(np.argmax([q[self.indices_gesto].max() for q in todas]))
        self.historico.append(todas[escolhida])
        self.media = np.mean(self.historico, axis=0)
        melhor = int(self.media.argmax())
        return self.nomes[melhor], float(self.media[melhor]), self.media

    @property
    def aquecendo(self):
        return len(self.buffer) < self.minimo


def pt(nome):
    return V.TRADUCAO.get(nome, nome) if nome else "—"


def tela(quadro, fontes, titulo, subtitulo, cor_titulo, progresso, cor_barra, info, placar, aviso=None, entrada=None):
    """Desenha a câmera espelhada com as instruções em cima, o placar à direita e o último resultado embaixo."""
    imagem = Image.fromarray(cv2.flip(quadro, 1)[..., ::-1]).convert("RGBA")
    camada = Image.new("RGBA", imagem.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(camada)
    L, A = imagem.size
    d.rectangle([0, 0, L, 118], fill=(0, 0, 0, 185))
    d.text((24, 14), titulo, font=fontes[0], fill=cor_titulo)
    d.text((26, 72), subtitulo, font=fontes[1], fill=BRANCO)
    if progresso is not None:
        d.rectangle([0, 112, L, 118], fill=(60, 60, 60, 255))
        d.rectangle([0, 112, L * max(0.0, min(1.0, progresso)), 118], fill=cor_barra)
    larg_placar = 330
    d.rectangle([L - larg_placar, 118, L, 118 + 44 + 26 * len(placar)], fill=(0, 0, 0, 150))
    d.text((L - larg_placar + 14, 126), "Placar desta sessão", font=fontes[2], fill=AMARELO)
    for i, (texto, valor, cor) in enumerate(placar):
        d.text((L - larg_placar + 14, 152 + 26 * i), texto, font=fontes[2], fill=cor)
        d.text((L - 16, 152 + 26 * i), valor, font=fontes[2], fill=cor, anchor="ra")
    d.rectangle([0, A - 70, L, A], fill=(0, 0, 0, 185))
    d.text((24, A - 52), info, font=fontes[2], fill=BRANCO)
    if aviso:
        texto, cor = aviso
        d.text((L / 2, A - 110), texto, font=fontes[0], fill=cor, anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0))
    final = Image.alpha_composite(imagem, camada).convert("RGB")
    if entrada is not None:  # o que o modelo vê: o quadrado do centro da câmera, sem espelhar
        mini = Image.fromarray(entrada.transpose(1, 2, 0).astype(np.uint8)).resize((150, 150), Image.NEAREST)
        final.paste(mini, (14, 132))
    return np.array(final)[..., ::-1].copy()


def resumir(tentativas, disparos_sem_gesto, minutos_sem_gesto):
    por_gesto = {}
    for t in tentativas:
        g = por_gesto.setdefault(t["gesto"], {"tentativas": 0, "acertos": 0, "erros": 0, "nao_disparou": 0, "latencias": [], "confundido_com": collections.Counter(),
                                              "com_posicao": 0, "top1": 0, "top5": 0})
        g["tentativas"] += 1
        if str(t.get("melhor_posicao", "")).strip():  # sessões antigas não anotavam a posição
            posicao = int(t["melhor_posicao"])
            g["com_posicao"] += 1
            g["top1"] += posicao == 1
            g["top5"] += posicao <= 5
        if t["resultado"] == "acerto":
            g["acertos"] += 1
            g["latencias"].append(float(t["latencia_s"]))
        elif t["resultado"] == "erro":
            g["erros"] += 1
            g["confundido_com"][t["disparou"]] += 1
        else:
            g["nao_disparou"] += 1
    saida = {}
    for nome, g in por_gesto.items():
        saida[nome] = {"gesto": pt(nome), "tentativas": g["tentativas"], "acertos": g["acertos"], "erros": g["erros"], "nao_disparou": g["nao_disparou"],
                       "taxa_acerto": round(g["acertos"] / g["tentativas"], 3),
                       "latencia_media_s": round(float(np.mean(g["latencias"])), 2) if g["latencias"] else None,
                       "confundido_com": {pt(k): v for k, v in g["confundido_com"].most_common()},
                       # Mesmo sem disparar: em quantas tentativas o gesto pedido chegou a 1º ou ao top 5 do ranking.
                       "com_posicao": g["com_posicao"], "top1": g["top1"], "top5": g["top5"]}
    total = len(tentativas)
    com_posicao = sum(g["com_posicao"] for g in saida.values())
    return {"gestos": saida, "tentativas": total, "acertos": sum(g["acertos"] for g in saida.values()),
            "ranking": {"tentativas_com_posicao": com_posicao, "top1": sum(g["top1"] for g in saida.values()),
                        "top5": sum(g["top5"] for g in saida.values())},
            "taxa_acerto": round(sum(g["acertos"] for g in saida.values()) / total, 3) if total else None,
            "sem_gesto": {"minutos": round(minutos_sem_gesto, 2), "falsos": disparos_sem_gesto,
                          "falsos_por_minuto": round(disparos_sem_gesto / minutos_sem_gesto, 2) if minutos_sem_gesto > 0 else None}}


def imprimir(resumo, titulo):
    print(f"\n=== {titulo} ===")
    print(f"{'Gesto':30s} {'acertos':>9s} {'taxa':>6s} {'tempo':>6s} {'top 5':>7s}  confundido com")
    for g in sorted(resumo["gestos"].values(), key=lambda x: -x["taxa_acerto"]):
        lat = f"{g['latencia_media_s']:.1f}s" if g["latencia_media_s"] is not None else "—"
        top5 = f"{g['top5']}/{g['com_posicao']}" if g["com_posicao"] else "—"
        conf = ", ".join(f"{k} ({v})" for k, v in g["confundido_com"].items())
        print(f"{g['gesto']:30s} {g['acertos']:>4d}/{g['tentativas']:<4d} {g['taxa_acerto']:>6.0%} {lat:>6s} {top5:>7s}  {conf}")
    if resumo["tentativas"]:
        print(f"Total: {resumo['acertos']}/{resumo['tentativas']} ({resumo['taxa_acerto']:.0%})")
    rk = resumo["ranking"]
    if rk["tentativas_com_posicao"]:
        n = rk["tentativas_com_posicao"]
        print(f"Ranking, sem exigir disparo ({n} tentativas anotadas): 1º lugar {rk['top1']}/{n} ({rk['top1'] / n:.0%}), top 5 {rk['top5']}/{n} ({rk['top5'] / n:.0%})")
    s = resumo["sem_gesto"]
    if s["minutos"]:
        print(f"Sem gesto: {s['falsos']} falsos em {s['minutos']:.1f} min ({s['falsos_por_minuto']:.2f} por minuto)")


def resumo_geral():
    """Junta as sessões de todos os integrantes (uma pasta por sessão em resultados/webcam), por pessoa e no total.

    Cada pessoa envia só a pasta da própria sessão; os arquivos gerais são refeitos aqui a partir delas,
    por isso ficam fora do git e nunca entram em conflito.
    """
    tentativas, por_pessoa, sessoes = [], {}, 0
    for pasta in sorted(p for p in PASTA.iterdir() if p.is_dir() and (p / "resumo.json").exists()):
        with (pasta / "tentativas.csv").open(encoding="utf-8-sig") as f:
            linhas = list(csv.DictReader(f))
        r = json.loads((pasta / "resumo.json").read_text(encoding="utf-8-sig"))
        p = por_pessoa.setdefault(r["pessoa"], {"tentativas": [], "falsos": 0, "minutos": 0.0, "sessoes": 0})
        p["tentativas"] += linhas
        p["falsos"] += r["sem_gesto"]["falsos"]
        p["minutos"] += r["sem_gesto"]["minutos"]
        p["sessoes"] += 1
        tentativas += linhas
        sessoes += 1
    if not sessoes:
        raise SystemExit(f"Nenhuma sessão em {PASTA}.")
    resumo = resumir(tentativas, sum(p["falsos"] for p in por_pessoa.values()), sum(p["minutos"] for p in por_pessoa.values()))
    resumo["sessoes"], resumo["pessoas"] = sessoes, sorted(por_pessoa)
    resumo["por_pessoa"] = {}
    for nome, p in sorted(por_pessoa.items()):
        r = resumir(p["tentativas"], p["falsos"], p["minutos"])
        resumo["por_pessoa"][nome] = {"sessoes": p["sessoes"], "tentativas": r["tentativas"], "acertos": r["acertos"], "taxa_acerto": r["taxa_acerto"],
                                      "sem_gesto": r["sem_gesto"], "taxa_por_gesto": {g["gesto"]: g["taxa_acerto"] for g in r["gestos"].values()}}
    (PASTA / "resumo_geral.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    if tentativas:
        with (PASTA / "todas_tentativas.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for t in tentativas for k in t)), restval="")
            w.writeheader()
            w.writerows(tentativas)
    imprimir(resumo, f"Todas as sessões ({sessoes}, pessoas: {', '.join(resumo['pessoas'])})")
    print("\nPor pessoa:")
    for nome, p in resumo["por_pessoa"].items():
        s = p["sem_gesto"]
        falsos = f"{s['falsos']} falsos em {s['minutos']:.1f} min" if s["minutos"] else "sem bloco sem gesto"
        taxa = f"{p['taxa_acerto']:.0%}" if p["taxa_acerto"] is not None else "—"
        print(f"  {nome:20s} {p['acertos']}/{p['tentativas']} ({taxa}), {falsos}, {p['sessoes']} sessão(ões)")
    print(f"\nSalvo em {PASTA / 'resumo_geral.json'} e {PASTA / 'todas_tentativas.csv'}")


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--pessoa", help="Só letras, números, - e _")
    a.add_argument("--conjunto", choices=sorted(CONJUNTOS), default="demos")
    a.add_argument("--tentativas", type=int, default=3, help="Tentativas por gesto")
    a.add_argument("--preparo", type=float, default=2.5, help="Segundos com a mão abaixada antes de cada tentativa")
    a.add_argument("--tempo", type=float, default=5.0, help="Segundos para fazer o gesto")
    a.add_argument("--sem-gesto", type=float, default=60.0, help="Segundos do bloco sem gesto no fim (0 desliga)")
    a.add_argument("--modelo", default=str(RAIZ / "resultados/jester/cnn_3d_mais_gestos/melhor.pt"))
    a.add_argument("--camera", type=int, default=0)
    a.add_argument("--limiar", type=float, default=0.7)
    a.add_argument("--seguidas", type=int, default=3)
    a.add_argument("--pausa", type=float, default=1.2)
    a.add_argument("--suavizar", type=int, default=2)
    a.add_argument("--intervalo", type=float, default=0.15)
    a.add_argument("--janelas", type=float, nargs="+", default=None)
    a.add_argument("--resumo", action="store_true", help="Só junta as sessões já feitas e mostra o resultado")
    args = a.parse_args()
    if args.resumo:
        resumo_geral()
        return
    if not args.pessoa or not re.fullmatch(r"[\w-]+", args.pessoa):
        raise SystemExit("Informe --pessoa com letras, números, - e _.")

    modelo = Modelo(args)
    gestos = [g for g in CONJUNTOS[args.conjunto] if g in modelo.nomes]
    faltando = [g for g in CONJUNTOS[args.conjunto] if g not in modelo.nomes]
    if faltando:
        print("Este modelo não tem:", ", ".join(pt(g) for g in faltando), flush=True)
    # Ordem embaralhada em cada rodada, para o cansaço e a luz não pesarem sempre no mesmo gesto.
    plano = []
    sorteio = random.Random()
    for rodada in range(1, args.tentativas + 1):
        ordem = gestos[:]
        sorteio.shuffle(ordem)
        plano += [(g, rodada) for g in ordem]

    captura = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not captura.isOpened():
        raise SystemExit(f"Não consegui abrir a câmera {args.camera}. Feche outros apps que a usem.")
    nome_janela = "Testar gestos na webcam"
    cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nome_janela, 1280, 720)
    fontes = (V.fonte(44), V.fonte(24), V.fonte(19))

    inicio_sessao = datetime.datetime.now()
    pasta = PASTA / f"{args.pessoa}_{inicio_sessao:%Y%m%d_%H%M%S}"
    tentativas, disparos = [], []
    falsos_sem_gesto, minutos_sem_gesto = 0, 0.0
    gatilho = Gatilho(args.limiar, args.seguidas, args.pausa)
    estado = {"nome": None, "conf": 0.0}
    aviso, fim_aviso = None, 0.0
    pausado = False

    def placar():
        linhas = []
        for g in gestos:
            feitas = [t for t in tentativas if t["gesto"] == g]
            acertos = sum(t["resultado"] == "acerto" for t in feitas)
            cor = CINZA if not feitas else VERDE if acertos == len(feitas) else VERMELHO if acertos == 0 else AMARELO
            linhas.append((pt(g), f"{acertos}/{len(feitas)}", cor))
        if minutos_sem_gesto > 0:
            linhas.append(("Falsos sem gesto", str(falsos_sem_gesto), AMARELO))
        return linhas

    def passo(fase, alvo=None):
        """Lê um quadro, roda o modelo e o gatilho; devolve o disparo (ou None) e a tecla apertada."""
        ok, quadro = captura.read()
        if not ok:
            raise SystemExit("A câmera parou de enviar imagens.")
        agora = time.time()
        r = modelo.receber(quadro, agora)
        disparo = None
        if r is not None:
            estado["nome"], estado["conf"] = r[0], r[1]
            disparo = gatilho.atualizar(r[0], r[1], agora)
            if disparo:
                disparos.append({"momento": datetime.datetime.now().isoformat(timespec="milliseconds"), "fase": fase,
                                 "pedido": alvo or "", "disparou": disparo, "confianca": round(r[1], 3)})
        return quadro, disparo, agora, cv2.waitKey(1) & 0xFF

    def mostrar(quadro, titulo, subtitulo, cor, progresso, cor_barra, alvo=None):
        nonlocal aviso
        if aviso and time.time() > fim_aviso:
            aviso = None
        if modelo.aquecendo:
            info = "Aquecendo o modelo..."
        else:
            info = f"Modelo vê: {pt(estado['nome'])} ({estado['conf']:.0%})"
            if alvo is not None and modelo.media is not None:
                info += f"   ·   confiança em {pt(alvo)}: {modelo.media[modelo.nomes.index(alvo)]:.0%}"
        info += "      [Espaço] pula  [R] refaz  [P] pausa  [Q] encerra"
        cv2.imshow(nome_janela, tela(quadro, fontes, titulo, subtitulo, cor, progresso, cor_barra, info, placar(), aviso,
                                     modelo.buffer[-1] if modelo.buffer else None))

    def fechou():
        return cv2.getWindowProperty(nome_janela, cv2.WND_PROP_VISIBLE) < 1

    def esperar_pausa():
        nonlocal pausado
        while pausado:
            quadro, _, _, tecla = passo("pausa")
            mostrar(quadro, "PAUSADO", "Aperte P para continuar", AMARELO, None, AMARELO)
            if tecla == ord("p"):
                pausado = False
            if tecla in (ord("q"), 27) or fechou():
                return False
        return True

    interrompido = False
    try:
        # Aquecimento: enche o buffer antes da primeira tentativa.
        while modelo.aquecendo:
            quadro, _, _, tecla = passo("aquecimento")
            mostrar(quadro, f"Olá, {args.pessoa}!", f"{len(gestos)} gestos × {args.tentativas} tentativas. Centralize-se na câmera.", BRANCO, None, BRANCO)
            if tecla in (ord("q"), 27) or fechou():
                interrompido = True
                break
        i = 0
        while i < len(plano) and not interrompido:
            alvo, rodada = plano[i]
            dica = DICAS.get(alvo, "")
            # Preparo: mão abaixada. Disparos aqui ficam anotados, mas não contam na tentativa.
            fim = time.time() + args.preparo
            while (resta := fim - time.time()) > 0:
                quadro, _, _, tecla = passo("preparo", alvo)
                mostrar(quadro, f"Prepare: {pt(alvo).upper()}", f"Mão abaixada. {dica}", AMARELO, 1 - resta / args.preparo, AMARELO, alvo)
                if tecla == ord("p"):
                    pausado = True
                if (tecla in (ord("q"), 27) or fechou()) or not esperar_pausa():
                    interrompido = True
                    break
            if interrompido:
                break
            # Gesto: vale o primeiro disparo dentro do tempo.
            inicio, resultado, disparou, latencia, conf_max, pulou, refazer = time.time(), "nao_disparou", "", None, 0.0, False, False
            melhor_posicao, ultima_media = None, None
            while (passado := time.time() - inicio) < args.tempo:
                quadro, disparo, agora, tecla = passo("gesto", alvo)
                if modelo.media is not None and modelo.media is not ultima_media:
                    ultima_media = modelo.media
                    conf_max = max(conf_max, float(modelo.media[modelo.nomes.index(alvo)]))
                    # Posição do gesto pedido no ranking da rede (1 = mais provável), a melhor durante a tentativa.
                    posicao = int((modelo.media > modelo.media[modelo.nomes.index(alvo)]).sum()) + 1
                    melhor_posicao = posicao if melhor_posicao is None else min(melhor_posicao, posicao)
                mostrar(quadro, f"AGORA: {pt(alvo).upper()}", f"{dica}   ({i + 1}/{len(plano)}, rodada {rodada})", VERDE, passado / args.tempo, VERDE, alvo)
                if disparo:
                    resultado = "acerto" if disparo == alvo else "erro"
                    disparou, latencia = disparo, agora - inicio
                    break
                if tecla == ord(" "):
                    pulou = True
                    break
                if tecla == ord("r"):
                    refazer = True
                    break
                if tecla == ord("p"):
                    pausado = True
                    if not esperar_pausa():
                        interrompido = True
                        break
                    inicio = time.time()  # a pausa não gasta o tempo do gesto
                if tecla in (ord("q"), 27) or fechou():
                    interrompido = True
                    break
            if interrompido:
                break
            if refazer:
                aviso, fim_aviso = ("Refazendo", AMARELO), time.time() + 1.0
                continue
            if pulou:
                aviso, fim_aviso = ("Pulado", CINZA), time.time() + 1.0
                i += 1
                continue
            tentativas.append({"pessoa": args.pessoa, "sessao": pasta.name, "gesto": alvo, "gesto_pt": pt(alvo), "rodada": rodada, "resultado": resultado,
                               "disparou": disparou, "disparou_pt": pt(disparou) if disparou else "", "latencia_s": round(latencia, 2) if latencia is not None else "",
                               "confianca_max_no_pedido": round(conf_max, 3), "melhor_posicao": melhor_posicao if melhor_posicao is not None else ""})
            if resultado == "acerto":
                aviso = (f"ACERTOU em {latencia:.1f} s", VERDE)
            elif resultado == "erro":
                aviso = (f"ERROU: virou {pt(disparou)}", VERMELHO)
            else:
                aviso = (f"Não disparou (máx. {conf_max:.0%})", CINZA)
            fim_aviso = time.time() + 1.6
            i += 1

        # Bloco sem gesto: todo disparo é falso.
        if not interrompido and args.sem_gesto > 0:
            inicio = time.time()
            while (passado := time.time() - inicio) < args.sem_gesto:
                quadro, disparo, _, tecla = passo("sem_gesto")
                if disparo:
                    falsos_sem_gesto += 1
                    aviso, fim_aviso = (f"Falso disparo: {pt(disparo)}", VERMELHO), time.time() + 1.2
                minutos_sem_gesto = passado / 60
                mostrar(quadro, "SEM GESTO: aja normalmente", "Fale, olhe em volta, coce o rosto, mexa as mãos sem fazer gestos",
                        BRANCO, passado / args.sem_gesto, (120, 160, 255))
                if tecla in (ord("q"), 27) or fechou():
                    break
    finally:
        captura.release()
        cv2.destroyAllWindows()

    if not tentativas and minutos_sem_gesto == 0:
        print("Nada foi medido; nada salvo.")
        return
    pasta.mkdir(parents=True, exist_ok=True)
    campos = list(tentativas[0].keys()) if tentativas else ["pessoa", "sessao", "gesto", "gesto_pt", "rodada", "resultado", "disparou",
                                                             "disparou_pt", "latencia_s", "confianca_max_no_pedido"]
    with (pasta / "tentativas.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(tentativas)
    with (pasta / "disparos.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["momento", "fase", "pedido", "disparou", "confianca"])
        w.writeheader()
        w.writerows(disparos)
    resumo = resumir(tentativas, falsos_sem_gesto, minutos_sem_gesto)
    caminho_modelo = Path(args.modelo).resolve()
    resumo.update({"pessoa": args.pessoa, "inicio": inicio_sessao.isoformat(timespec="seconds"),
                   "modelo": caminho_modelo.relative_to(RAIZ).as_posix() if caminho_modelo.is_relative_to(RAIZ) else args.modelo,
                   "conjunto": args.conjunto, "interrompido": interrompido,
                   "regra": {"limiar": args.limiar, "seguidas": args.seguidas, "pausa_s": args.pausa, "janelas_s": modelo.janelas, "tempo_gesto_s": args.tempo}})
    (pasta / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    imprimir(resumo, f"Sessão de {args.pessoa}" + (" (interrompida)" if interrompido else ""))
    print(f"\nSalvo em {pasta}. Para juntar todas as sessões: python demo/testar_webcam.py --resumo")


if __name__ == "__main__":
    main()
