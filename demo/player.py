"""Player controlado por gestos: o modelo roda aqui e a interface abre no navegador.

Arrastar a mão para a esquerda volta uma música, para a direita passa para a próxima, e o sinal
de pare alterna play e pause. Tudo fica neste computador: o servidor só atende 127.0.0.1.

Uso:
    python demo/player.py
    python demo/player.py --limiar 0.7 --seguidas 3 --camera 1
"""
import argparse
import collections
import json
import queue
import re
import os
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ao_vivo as V

RAIZ = Path(__file__).resolve().parent
COMANDOS = {"Swiping Left": "anterior", "Swiping Right": "proxima", "Stop Sign": "alternar"}
COMANDOS_VOLUME = {"Thumb Up": "volume_mais", "Thumb Down": "volume_menos"}
COMANDOS_PAINEL = {**COMANDOS_VOLUME, "Drumming Fingers": "tamborilar", "Pushing Hand Away": "ajuda",
                   # Um dedo só existe no modelo cnn_3d_mais_gestos e avança a página; puxar a mão volta.
                   # Dois a quatro dedos e o punho ficaram de fora: na webcam, quase nunca passavam do limiar.
                   "One Finger": "funcao", "Pulling Hand In": "funcao_anterior",
                   "Two Fingers": "tamborilar"}  # dois dedos faz o mesmo que tamborilar: abre e fecha o menu
# No hospital, empurrar a mão fica de fora: o modelo a confunde com o zoom in.
COMANDOS_HOSPITAL = {"Thumb Up": "confirmar", "Thumb Down": "desfazer", "Shaking Hand": "normal",
                     "Zooming In With Full Hand": "zoom_mais", "Zooming Out With Full Hand": "zoom_menos",
                     # Só existem no modelo cnn_3d_mais_gestos.
                     "One Finger": "proximo_corte", "Two Fingers": "janela"}
# Na apresentação, a página repassa cada gesto para o slide aberto. O servidor só dispara gestos desta
# lista e manda o nome da classe; quem traduz para o comando de cada demo é a própria apresentação.
# Carro 3D (só Música, Ar-condicionado e Telefone). Os números vão direto para uma página (só com --pontos, que reconhece
# os dedos). Três dedos fica de fora: o OK, gesto de ativação reconhecido na página, tem três dedos esticados.
# O sinal de pare e empurrar a mão fazem o mesmo (tocar/pausar, ligar/desligar, confirmar); puxar a mão explica.
COMANDOS_CARRO3D = {"Swiping Left": "anterior", "Swiping Right": "proxima", **COMANDOS_VOLUME, "Drumming Fingers": "tamborilar",
                    "Stop Sign": "alternar", "Pushing Hand Away": "alternar", "Pulling Hand In": "ajuda",
                    "One Finger": "pagina_1", "Two Fingers": "pagina_2", "Four Fingers": "pagina_4"}
GESTOS_APRESENTACAO = sorted({*COMANDOS, *COMANDOS_PAINEL, *COMANDOS_HOSPITAL, *COMANDOS_CARRO3D, "Rolling Hand Forward", "Rolling Hand Backward"})
# Páginas que a apresentação pode abrir dentro de um slide.
DEMOS_EMBUTIDAS = {"tenis.html", "painel_carro.html", "hospital.html", "hand3d.html", "carro3d.html"}
# Modelo do MediaPipe usado pela mão 3D (demo/hand3d.html), servido de modelos_externos/.
MODELO_MAO = RAIZ.parent / "modelos_externos" / "hand_landmarker.task"
TIPOS_AUDIO = {".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".wav": "audio/wav", ".ogg": "audio/ogg"}


class Estado:
    def __init__(self):
        self.trava = threading.Lock()
        self.jpeg = None
        self.jpeg_cru = None  # quadro maior e sem espelhar, para o MediaPipe do navegador (mãos 3D)
        self.quer_cru = 0     # quantas páginas estão lendo /video_cru
        self.assinantes = []
        self.inicio = time.time()
        self.disparos = 0
        # Ligado pela mão 3D da apresentação: o laço solta a webcam para o navegador usar e só a pega de
        # volta quando o evento é desligado (a câmera do Windows não pode ser aberta por dois programas).
        self.soltar_camera = threading.Event()

    def publicar(self, evento):
        dados = json.dumps(evento, ensure_ascii=False)
        with self.trava:
            for fila in self.assinantes:
                try:
                    fila.put_nowait(dados)
                except queue.Full:
                    pass


class Gatilho:
    """Só dispara com a mesma classe em N janelas seguidas e exige soltar o gesto para repetir.

    Os gestos de `repetiveis` são a exceção: mantidos, voltam a disparar a cada `repetir` segundos.
    A repetição exige o gesto na janela mais curta (`presente`): as janelas longas ainda enxergam um
    gesto rápido alguns segundos depois de a mão sair, o que contaria como se ele continuasse.
    """

    def __init__(self, limiar, seguidas, pausa, repetiveis=(), repetir=2.0):
        self.limiar, self.seguidas, self.pausa = limiar, seguidas, pausa
        self.repetiveis, self.repetir = set(repetiveis), repetir
        self.atual, self.contagem, self.fim_pausa, self.armado = None, 0, 0.0, True
        self.proxima_repeticao, self.foi_repeticao = 0.0, False

    def atualizar(self, nome, confianca, agora, presente=True):
        if nome not in COMANDOS or confianca < self.limiar:
            self.atual, self.contagem, self.armado = None, 0, True
            return None
        if nome != self.atual:
            self.atual, self.contagem, self.armado = nome, 0, True
        self.contagem += 1
        self.foi_repeticao = False
        if self.armado and self.contagem >= self.seguidas and agora >= self.fim_pausa:
            self.armado, self.fim_pausa = False, agora + self.pausa
            self.proxima_repeticao = agora + self.repetir
            return nome
        if not self.armado and presente and nome in self.repetiveis and agora >= self.proxima_repeticao:
            self.proxima_repeticao, self.fim_pausa = agora + self.repetir, agora + self.pausa
            self.foi_repeticao = True
            return nome
        return None


def abrir_tela_cheia(endereco):
    """Abre o Chrome (ou o Edge) em tela cheia, numa janela própria; sem eles, usa o navegador padrão."""
    candidatos = [Path(os.environ.get(v, "")) / sub for v in ("ProgramFiles", "ProgramFiles(x86)", "LocalAppData")
                  for sub in ("Google/Chrome/Application/chrome.exe", "Microsoft/Edge/Application/msedge.exe")]
    navegador = next((c for c in candidatos if c.is_file()), None)
    if navegador is None:
        webbrowser.open(endereco)
        return
    # Perfil separado: assim a janela abre mesmo com o navegador já aberto, e respeita a tela cheia.
    perfil = Path(tempfile.gettempdir()) / "gestos_apresentacao"
    subprocess.Popen([str(navegador), "--start-fullscreen", f"--app={endereco}", f"--user-data-dir={perfil}",
                      "--no-first-run", "--autoplay-policy=no-user-gesture-required"])


def publicar_quadros(estado, quadro, agora):
    """Miniatura espelhada (/video, a cada ~0,1 s) e quadro maior sem espelhar (/video_cru, a cada 0,05 s)."""
    if agora - getattr(estado, "_ultima_mini", 0) >= 0.08:
        pequena = cv2.resize(cv2.flip(quadro, 1), (480, 270), interpolation=cv2.INTER_AREA)
        ok, codificado = cv2.imencode(".jpg", pequena, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            estado.jpeg = codificado.tobytes()
        estado._ultima_mini = agora
    if estado.quer_cru > 0:  # só codifica o quadro grande enquanto alguma página o usa
        a, l = quadro.shape[:2]
        grande = cv2.resize(quadro, (960, round(a * 960 / l)), interpolation=cv2.INTER_AREA) if l > 960 else quadro
        ok, codificado = cv2.imencode(".jpg", grande, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            estado.jpeg_cru = codificado.tobytes()


def abrir_camera(args):
    captura = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return captura


def esperar_camera(args, estado, parar, captura):
    """Solta a webcam enquanto a mão 3D a usa e a abre de novo depois. Devolve a nova captura (ou None se falhar)."""
    captura.release()
    estado.publicar({"tipo": "camera", "estado": "solta"})
    while estado.soltar_camera.is_set() and not parar.is_set():
        time.sleep(0.1)
    for _ in range(20):  # o navegador pode levar um instante para liberar a câmera
        nova = abrir_camera(args)
        if nova.isOpened():
            estado.publicar({"tipo": "camera", "estado": "retomada"})
            return nova
        nova.release()
        time.sleep(0.3)
    estado.publicar({"tipo": "erro", "mensagem": "Não consegui abrir a câmera de novo depois da mão 3D."})
    return None


class Pontos:
    """Pontos da mão (MediaPipe) + GRU, combinados com a CNN como em src/fluxo_combinado.py.

    A cada quadro do buffer da CNN (12 por segundo), o MediaPipe acha a mão no quadro inteiro. Em cada análise, a GRU
    lê os pontos das mesmas janelas da CNN. Nos números (1 a 4 dedos) decide a
    GRU, com a mão parada; nos outros, a CNN (ver combinar).
    """

    def __init__(self, args, nomes, dispositivo):
        import pontos_mao as P
        import treinar_pontos_jester as TP
        self.TP, self.dispositivo = TP, dispositivo
        e = torch.load(args.pontos, map_location=dispositivo, weights_only=False)
        if list(e["nomes"]) != list(nomes):
            raise SystemExit("A GRU e a CNN precisam ter as mesmas classes, na mesma ordem.")
        self.gru = TP.Rede(e["entrada"], len(nomes)).to(dispositivo).eval()
        self.gru.load_state_dict(e["modelo"])
        self.media, self.desvio = e["media"], e["desvio"]
        self.nomes = list(nomes)
        self.numeros = np.array([nomes.index(n) for n in ("One Finger", "Two Fingers", "Three Fingers", "Four Fingers") if n in nomes])
        self.detector = P.DetectorMao(video=True)
        self.ultimo = {"mao": False, "numero": None, "nota": 0.0, "parada": False}
        print(f"Pontos da mão + GRU ligados ({Path(args.pontos).name}): números pela GRU (com a mão parada), o resto pela CNN", flush=True)

    def quadro(self, quadro, agora):
        """Pontos da mão principal de um quadro, no formato da GRU."""
        mao = self.detector.detectar(quadro, agora)
        img, world = np.zeros((2, 21, 3), np.float32), np.zeros((2, 21, 3), np.float32)
        pres, score = np.zeros(2, bool), np.zeros(2, np.float32)
        if mao is not None:
            img[0], world[0], pres[0], score[0] = mao.imagem, mao.mundo, True, mao.confianca
        return img, world, pres, score, quadro.shape[:2]

    def mao_parada(self, lista, n=6):
        """A mão apareceu em quase todos os últimos `n` quadros (0,5 s) e o pulso andou menos de meia palma."""
        q = lista[-n:]
        pres = [x[2][0] for x in q]
        if sum(pres) < n - 1:
            return False
        a, l = q[-1][4]
        xy = [(x[0][0, :, :2] * np.array([l, a], np.float32)) for x, ok in zip(q, pres) if ok]
        palma = max(float(np.linalg.norm(xy[-1][9] - xy[-1][0])), 1e-6)
        return max(float(np.linalg.norm(t[0] - xy[-1][0])) for t in xy) / palma <= 0.5

    @torch.no_grad()
    def combinar(self, notas_cnn, lista, janelas):
        """notas_cnn: [janelas, classes], na ordem de `janelas` (em quadros). Devolve as notas combinadas.

        A GRU diz se é um número (a soma das notas dela nos números) e qual é. Se for, a nota de cada número é a da GRU;
        o resto (1 - essa soma) fica com a CNN, sem as classes de número. Com a mão em movimento, número não vale.
        No teste misto: 91,2% de acerto (Jester 93,1%, números 95,7%) e 97% dos números certos com nota >= 0,7.
        """
        xs, ms = [], []
        for n in janelas:
            q = lista[-n:]
            c = {"img": np.stack([x[0] for x in q]), "world": np.stack([x[1] for x in q]), "present": np.stack([x[2] for x in q]),
                 "score": np.stack([x[3] for x in q]), "altura_largura": q[-1][4]}
            x = self.TP.atributos(c)[0]
            x = np.where(x[:, -2:-1] > 0, (x - self.media) / self.desvio, 0).astype(np.float32)
            a, m = self.TP.fixar(x)
            xs.append(a)
            ms.append(m)
        notas_gru = self.gru(torch.from_numpy(np.stack(xs)).to(self.dispositivo),
                             torch.from_numpy(np.stack(ms)).to(self.dispositivo)).float().softmax(1).cpu().numpy()
        parada = self.mao_parada(lista)
        num = self.numeros
        pn = notas_gru[:, num].sum(1, keepdims=True) if parada else np.zeros((len(notas_gru), 1), np.float32)
        outros = notas_cnn.copy()
        outros[:, num] = 0
        outros /= np.maximum(outros.sum(1, keepdims=True), 1e-9)
        p = (1 - pn) * outros
        if parada:
            p[:, num] = notas_gru[:, num]
        k = int(notas_gru[0, num].argmax())  # janela mais curta
        self.ultimo = {"mao": bool(lista[-1][2][0]), "parada": parada, "numero": self.nomes[num[k]], "nota": float(notas_gru[0, num[k]])}
        return p


def laco_modelo(args, estado, parar):
    dispositivo = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    # Sem NVIDIA, cada análise leva cerca de 0,4 s em vez de uma fração disso. Com 3 janelas
    # seguidas, um gesto rápido acabaria antes de completar o gatilho; com 2 trechos em vez de 3,
    # cada análise sai mais rápida. Valores passados no comando, como no tênis, têm prioridade.
    if args.seguidas is None:
        args.seguidas = 3 if dispositivo == "cuda" else 2
    if args.janelas is None:
        args.janelas = [1.0, 1.5, 3.0] if dispositivo == "cuda" else [1.0, 1.5]
    caminho = Path(args.modelo) if args.modelo else next((p for p in V.MODELOS_PADRAO if p.exists()), None)
    if caminho is None or not caminho.exists():
        raise SystemExit("Nenhum modelo treinado encontrado.")
    modelo, nomes, info = V.carregar_modelo(caminho, dispositivo)
    validacao = f" (validação {info['top1_val']:.1%})" if "top1_val" in info else ""
    print(f"Modelo: {caminho.parent.name}/{caminho.name}{validacao}, {dispositivo}", flush=True)
    indices_gesto = [i for i, n in enumerate(nomes) if n not in V.SEM_GESTO]
    pontos = Pontos(args, nomes, dispositivo) if args.pontos else None

    captura = None
    if not args.simular:
        captura = abrir_camera(args)
        if not captura.isOpened():
            print(f"ERRO: não consegui abrir a câmera {args.camera}. Feche outros apps que a usem.", flush=True)
            estado.publicar({"tipo": "erro", "mensagem": "Não consegui abrir a câmera. Feche outros apps que a usem."})
            return

    if dispositivo == "cpu":
        # Sem NVIDIA, câmera e modelo rodam separados. Com NVIDIA, o laço abaixo segue igual.
        if pontos is not None:
            print("Aviso: sem NVIDIA, os pontos da mão ficam desligados (só a CNN).", flush=True)
        laco_separado(args, estado, parar, captura, modelo, nomes, indices_gesto)
        return

    if pontos is not None:
        laco_pontos(args, estado, parar, captura, modelo, nomes, indices_gesto, pontos, dispositivo)
        return

    maximo = int(round(max(args.janelas) * V.QUADROS_POR_SEGUNDO))
    minimo = int(round(min(args.janelas) * V.QUADROS_POR_SEGUNDO))
    buffer, historico = collections.deque(maxlen=maximo), collections.deque(maxlen=args.suavizar)
    buffer_pontos = collections.deque(maxlen=maximo)  # pontos da mão de cada quadro do buffer (com --pontos)
    gatilho = Gatilho(args.limiar, args.seguidas, args.pausa, COMANDOS_VOLUME if (args.painel or args.apresentacao) else (), args.repetir)
    ultimo_quadro, ultima_inferencia, ultimo_video, ultimo_estado = 0.0, 0.0, 0.0, 0.0
    tempos = collections.deque(maxlen=30)
    nome_atual, confianca, ms = "Aquecendo", 0.0, 0.0

    while not parar.is_set():
        if captura is not None and estado.soltar_camera.is_set():
            captura = esperar_camera(args, estado, parar, captura)
            buffer.clear(); historico.clear(); buffer_pontos.clear()
            if captura is None:
                break
            continue
        if captura is None:
            quadro = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
            time.sleep(0.03)
        else:
            ok, quadro = captura.read()
            if not ok:
                estado.publicar({"tipo": "erro", "mensagem": "A câmera parou de enviar imagens."})
                break
        agora = time.time()
        tempos.append(agora)
        if agora - ultimo_quadro >= 1 / V.QUADROS_POR_SEGUNDO:
            buffer.append(V.preparar_quadro(quadro))
            if pontos is not None:
                buffer_pontos.append(pontos.quadro(quadro, agora))
            ultimo_quadro = agora
        if len(buffer) >= minimo and agora - ultima_inferencia >= args.intervalo:
            t0 = time.perf_counter()
            lista, lotes = list(buffer), []
            for duracao in args.janelas:
                n = int(round(duracao * V.QUADROS_POR_SEGUNDO))
                if len(lista) >= n:
                    indices = np.round(np.linspace(len(lista) - n, len(lista) - 1, V.N_FRAMES)).astype(int)
                    lotes.append(np.stack([lista[i] for i in indices]))
            with torch.no_grad(), torch.autocast(dispositivo, dtype=torch.bfloat16):
                x = (torch.from_numpy(np.stack(lotes)).to(dispositivo) - 127.5) / 127.5
                todas = modelo(x).float().softmax(1).cpu().numpy()
            if pontos is not None:
                usadas = [int(round(d * V.QUADROS_POR_SEGUNDO)) for d in args.janelas if len(lista) >= int(round(d * V.QUADROS_POR_SEGUNDO))]
                todas = pontos.combinar(todas, list(buffer_pontos), usadas)
            escolhida = int(np.argmax([q[indices_gesto].max() for q in todas]))
            historico.append(todas[escolhida])
            media = np.mean(historico, axis=0)
            melhor = int(media.argmax())
            nome_atual, confianca = nomes[melhor], float(media[melhor])
            ms = (time.perf_counter() - t0) * 1000
            ultima_inferencia = agora
            curta = todas[0]  # lotes segue a ordem de --janelas, da mais curta para a mais longa
            presente = nomes[int(curta.argmax())] == nome_atual and float(curta.max()) >= args.limiar
            disparo = gatilho.atualizar(nome_atual, confianca, agora, presente)
            if disparo and disparo in COMANDOS:
                estado.disparos += 1
                estado.publicar({"tipo": "comando", "comando": COMANDOS[disparo], "classe": disparo, "gesto": V.TRADUCAO.get(disparo, disparo),
                                 "confianca": confianca, "repeticao": gatilho.foi_repeticao})
        if agora - ultimo_video >= 0.05:
            publicar_quadros(estado, quadro, agora)
            ultimo_video = agora
        if agora - ultimo_estado >= 0.2:
            fps = (len(tempos) - 1) / max(tempos[-1] - tempos[0], 1e-3)
            minutos = max((agora - estado.inicio) / 60, 1e-3)
            estado.publicar({"tipo": "estado", "gesto": V.TRADUCAO.get(nome_atual, nome_atual),
                             "e_comando": nome_atual in COMANDOS, "confianca": confianca, "disparos": estado.disparos,
                             "por_minuto": estado.disparos / minutos, "fps": fps, "ms": ms,
                             "aquecendo": len(buffer) < minimo})
            ultimo_estado = agora
    if captura is not None:
        captura.release()


def laco_pontos(args, estado, parar, captura, modelo, nomes, indices_gesto, pontos, dispositivo):
    """Laço do modelo final (CNN + pontos da mão + GRU), com NVIDIA.

    A câmera lê numa thread própria e guarda só o quadro mais recente; este laço pega um quadro a cada 1/12 s, roda o
    MediaPipe nele e, a cada --intervalo, a CNN e a GRU. Assim a câmera fica a ~30 quadros/s e o modelo recebe 12
    quadros por segundo de verdade, como no treino.
    """
    maximo = int(round(max(args.janelas) * V.QUADROS_POR_SEGUNDO))
    minimo = int(round(min(args.janelas) * V.QUADROS_POR_SEGUNDO))
    buffer, buffer_pontos = collections.deque(maxlen=maximo), collections.deque(maxlen=maximo)
    historico = collections.deque(maxlen=args.suavizar)
    trava = threading.Lock()
    camera = {"quadro": None, "fps": 0.0, "parou": False, "novo": False}

    def ler_camera():
        nonlocal captura
        tempos = collections.deque(maxlen=30)
        ultimo_video = 0.0
        while not parar.is_set():
            if captura is not None and estado.soltar_camera.is_set():
                captura = esperar_camera(args, estado, parar, captura)
                with trava:
                    buffer.clear(); buffer_pontos.clear()
                if captura is None:
                    camera["parou"] = True
                    return
                continue
            if captura is None:
                quadro = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
                time.sleep(0.03)
            else:
                ok, quadro = captura.read()
                if not ok:
                    estado.publicar({"tipo": "erro", "mensagem": "A câmera parou de enviar imagens."})
                    camera["parou"] = True
                    return
            agora = time.time()
            tempos.append(agora)
            camera["fps"] = (len(tempos) - 1) / max(tempos[-1] - tempos[0], 1e-3)
            with trava:
                camera["quadro"], camera["novo"] = quadro, True
            if agora - ultimo_video >= 0.05:
                publicar_quadros(estado, quadro, agora)
                ultimo_video = agora

    leitor = threading.Thread(target=ler_camera, daemon=True)
    leitor.start()
    gatilho = Gatilho(args.limiar, args.seguidas, args.pausa, COMANDOS_VOLUME if (args.painel or args.apresentacao) else (), args.repetir)
    proximo_quadro, ultima_inferencia, ultimo_estado = time.time(), 0.0, 0.0
    nome_atual, confianca, ms = "Aquecendo", 0.0, 0.0
    passo = 1 / V.QUADROS_POR_SEGUNDO
    while not parar.is_set() and not camera["parou"]:
        agora = time.time()
        if agora < proximo_quadro:
            time.sleep(min(0.005, proximo_quadro - agora))
            continue
        proximo_quadro = max(proximo_quadro + passo, agora - passo)  # sem acumular atraso
        with trava:
            quadro = camera["quadro"]
        if quadro is None:
            continue
        preparado, marcado = V.preparar_quadro(quadro), pontos.quadro(quadro, agora)
        with trava:
            buffer.append(preparado); buffer_pontos.append(marcado)
            lista, lista_pontos = list(buffer), list(buffer_pontos)
        if len(lista) >= minimo and agora - ultima_inferencia >= args.intervalo:
            t0 = time.perf_counter()
            lotes, usadas = [], []
            for duracao in args.janelas:
                n = int(round(duracao * V.QUADROS_POR_SEGUNDO))
                if len(lista) >= n:
                    indices = np.round(np.linspace(len(lista) - n, len(lista) - 1, V.N_FRAMES)).astype(int)
                    lotes.append(np.stack([lista[i] for i in indices]))
                    usadas.append(n)
            with torch.no_grad(), torch.autocast(dispositivo, dtype=torch.bfloat16):
                x = (torch.from_numpy(np.stack(lotes)).to(dispositivo) - 127.5) / 127.5
                todas = modelo(x).float().softmax(1).cpu().numpy()
            todas = pontos.combinar(todas, lista_pontos, usadas)
            escolhida = int(np.argmax([q[indices_gesto].max() for q in todas]))
            historico.append(todas[escolhida])
            media = np.mean(historico, axis=0)
            melhor = int(media.argmax())
            nome_atual, confianca = nomes[melhor], float(media[melhor])
            ms = (time.perf_counter() - t0) * 1000
            ultima_inferencia = agora
            curta = todas[0]
            presente = nomes[int(curta.argmax())] == nome_atual and float(curta.max()) >= args.limiar
            disparo = gatilho.atualizar(nome_atual, confianca, agora, presente)
            if disparo and disparo in COMANDOS:
                estado.disparos += 1
                estado.publicar({"tipo": "comando", "comando": COMANDOS[disparo], "classe": disparo, "gesto": V.TRADUCAO.get(disparo, disparo),
                                 "confianca": confianca, "repeticao": gatilho.foi_repeticao})
        if agora - ultimo_estado >= 0.2:
            minutos = max((agora - estado.inicio) / 60, 1e-3)
            u = pontos.ultimo
            estado.publicar({"tipo": "estado", "gesto": V.TRADUCAO.get(nome_atual, nome_atual),
                             "e_comando": nome_atual in COMANDOS, "confianca": confianca, "disparos": estado.disparos,
                             "por_minuto": estado.disparos / minutos, "fps": camera["fps"], "ms": ms,
                             "aquecendo": len(lista) < minimo,
                             "pontos": {"mao": u["mao"], "parada": u["parada"], "numero": V.TRADUCAO.get(u["numero"], u["numero"]), "nota": u["nota"]}})
            ultimo_estado = agora
    leitor.join(timeout=2)
    if captura is not None:
        captura.release()


def laco_separado(args, estado, parar, captura, modelo, nomes, indices_gesto):
    """Versão do laço para quando o modelo roda no processador, sem NVIDIA.

    No processador, cada análise leva cerca de meio segundo. No laço único de laco_modelo, a câmera
    esperaria cada análise terminar: a imagem congelaria e o buffer receberia menos quadros que os
    12 por segundo que o modelo espera. Aqui a câmera lê em uma thread própria e nunca espera o
    modelo; o modelo analisa os quadros mais recentes assim que termina a análise anterior.

    O autocast em bfloat16 fica de fora: em processadores sem suporte a ele, uma única análise
    passou de 4 minutos, contra cerca de 0,5 s em float32.
    """
    maximo = int(round(max(args.janelas) * V.QUADROS_POR_SEGUNDO))
    minimo = int(round(min(args.janelas) * V.QUADROS_POR_SEGUNDO))
    buffer = collections.deque(maxlen=maximo)
    trava_buffer = threading.Lock()  # a câmera acrescenta quadros enquanto o modelo copia o buffer
    camera = {"fps": 0.0, "parou": False}

    def ler_camera():
        nonlocal captura
        tempos = collections.deque(maxlen=30)
        ultimo_quadro, ultimo_video = 0.0, 0.0
        while not parar.is_set():
            if captura is not None and estado.soltar_camera.is_set():
                captura = esperar_camera(args, estado, parar, captura)
                with trava_buffer:
                    buffer.clear()
                if captura is None:
                    camera["parou"] = True
                    return
                continue
            if captura is None:
                quadro = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
                time.sleep(0.03)
            else:
                ok, quadro = captura.read()
                if not ok:
                    estado.publicar({"tipo": "erro", "mensagem": "A câmera parou de enviar imagens."})
                    camera["parou"] = True
                    return
            agora = time.time()
            tempos.append(agora)
            camera["fps"] = (len(tempos) - 1) / max(tempos[-1] - tempos[0], 1e-3)
            if agora - ultimo_quadro >= 1 / V.QUADROS_POR_SEGUNDO:
                preparado = V.preparar_quadro(quadro)
                with trava_buffer:
                    buffer.append(preparado)
                ultimo_quadro = agora
            if agora - ultimo_video >= 0.05:
                publicar_quadros(estado, quadro, agora)
                ultimo_video = agora

    leitor = threading.Thread(target=ler_camera, daemon=True)
    leitor.start()

    historico = collections.deque(maxlen=args.suavizar)
    gatilho = Gatilho(args.limiar, args.seguidas, args.pausa, COMANDOS_VOLUME if (args.painel or args.apresentacao) else (), args.repetir)
    ultima_inferencia, ultimo_estado = 0.0, 0.0
    nome_atual, confianca, ms = "Aquecendo", 0.0, 0.0

    while not parar.is_set() and not camera["parou"]:
        agora = time.time()
        with trava_buffer:
            lista = list(buffer)
        if len(lista) >= minimo and agora - ultima_inferencia >= args.intervalo:
            t0 = time.perf_counter()
            lotes = []
            for duracao in args.janelas:
                n = int(round(duracao * V.QUADROS_POR_SEGUNDO))
                if len(lista) >= n:
                    indices = np.round(np.linspace(len(lista) - n, len(lista) - 1, V.N_FRAMES)).astype(int)
                    lotes.append(np.stack([lista[i] for i in indices]))
            with torch.no_grad():
                x = (torch.from_numpy(np.stack(lotes)) - 127.5) / 127.5
                todas = modelo(x).float().softmax(1).numpy()
            escolhida = int(np.argmax([q[indices_gesto].max() for q in todas]))
            historico.append(todas[escolhida])
            media = np.mean(historico, axis=0)
            melhor = int(media.argmax())
            nome_atual, confianca = nomes[melhor], float(media[melhor])
            ms = (time.perf_counter() - t0) * 1000
            ultima_inferencia = agora
            curta = todas[0]  # lotes segue a ordem de --janelas, da mais curta para a mais longa
            presente = nomes[int(curta.argmax())] == nome_atual and float(curta.max()) >= args.limiar
            disparo = gatilho.atualizar(nome_atual, confianca, agora, presente)
            if disparo:
                estado.disparos += 1
                estado.publicar({"tipo": "comando", "comando": COMANDOS[disparo], "classe": disparo, "gesto": V.TRADUCAO[disparo],
                                 "confianca": confianca, "repeticao": gatilho.foi_repeticao})
        else:
            time.sleep(0.01)
        if agora - ultimo_estado >= 0.2:
            minutos = max((agora - estado.inicio) / 60, 1e-3)
            estado.publicar({"tipo": "estado", "gesto": V.TRADUCAO.get(nome_atual, nome_atual),
                             "e_comando": nome_atual in COMANDOS, "confianca": confianca, "disparos": estado.disparos,
                             "por_minuto": estado.disparos / minutos, "fps": camera["fps"], "ms": ms,
                             "aquecendo": len(lista) < minimo})
            ultimo_estado = agora
    leitor.join(timeout=2)
    if captura is not None:
        captura.release()


def criar_servidor(args, estado, parar):
    html = (RAIZ / args.pagina).read_bytes()

    class Atendente(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def enviar_audio(self, arquivo):
            """Entrega o áudio, inclusive em partes, para o navegador poder avançar na faixa."""
            if arquivo.suffix.lower() not in TIPOS_AUDIO or not arquivo.is_file():
                self.send_response(404)
                self.end_headers()
                return
            tamanho = arquivo.stat().st_size
            inicio, fim, parcial = 0, tamanho - 1, False
            faixa = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if faixa and (faixa[1] or faixa[2]):
                parcial = True
                if faixa[1]:
                    inicio = int(faixa[1])
                    fim = min(int(faixa[2]), tamanho - 1) if faixa[2] else tamanho - 1
                else:
                    inicio = max(tamanho - int(faixa[2]), 0)
                if inicio > fim:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{tamanho}")
                    self.end_headers()
                    return
            self.send_response(206 if parcial else 200)
            self.send_header("Content-Type", TIPOS_AUDIO[arquivo.suffix.lower()])
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(fim - inicio + 1))
            if parcial:
                self.send_header("Content-Range", f"bytes {inicio}-{fim}/{tamanho}")
            self.end_headers()
            try:
                with arquivo.open("rb") as f:
                    f.seek(inicio)
                    restante = fim - inicio + 1
                    while restante > 0 and (bloco := f.read(min(65536, restante))):
                        self.wfile.write(bloco)
                        restante -= len(bloco)
            except OSError:
                pass

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(html)
            elif url.path.startswith("/demo/") and Path(url.path).name in DEMOS_EMBUTIDAS:
                # A apresentação abre as demos dentro de slides; lidas a cada pedido, sem cache.
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write((RAIZ / Path(url.path).name).read_bytes())
            elif url.path.startswith("/vendor/"):
                # Só arquivos diretos de demo/vendor, sem subpastas.
                arquivo = RAIZ / "vendor" / Path(url.path).name
                if not arquivo.is_file():
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript; charset=utf-8")
                self.end_headers()
                self.wfile.write(arquivo.read_bytes())
            elif url.path in ("/camera/soltar", "/camera/retomar"):
                # A mão 3D (demo/hand3d.html) pede a webcam emprestada e a devolve ao sair do slide.
                (estado.soltar_camera.set if url.path.endswith("soltar") else estado.soltar_camera.clear)()
                self.send_response(204)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
            elif url.path == "/modelos_externos/hand_landmarker.task":
                if not MODELO_MAO.is_file():
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(MODELO_MAO.stat().st_size))
                self.end_headers()
                self.wfile.write(MODELO_MAO.read_bytes())
            elif url.path == "/musicas/lista":
                pasta = Path(args.musicas)
                nomes = sorted(a.name for a in pasta.iterdir() if a.suffix.lower() in TIPOS_AUDIO) if pasta.is_dir() else []
                corpo = json.dumps(nomes, ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(corpo)
            elif url.path.startswith("/musicas/"):
                self.enviar_audio(Path(args.musicas) / Path(unquote(url.path)).name)
            elif url.path == "/eventos":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                fila = queue.Queue(maxsize=50)
                with estado.trava:
                    estado.assinantes.append(fila)
                try:
                    while not parar.is_set():
                        try:
                            self.wfile.write(f"data: {fila.get(timeout=5)}\n\n".encode())
                        except queue.Empty:
                            self.wfile.write(b": ativo\n\n")
                        self.wfile.flush()
                except OSError:
                    pass
                finally:
                    with estado.trava:
                        estado.assinantes.remove(fila)
            elif url.path == "/video":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=quadro")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    while not parar.is_set():
                        jpeg = estado.jpeg
                        if jpeg:
                            self.wfile.write(b"--quadro\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                            self.wfile.flush()
                        time.sleep(0.08)
                except OSError:
                    pass
            elif url.path == "/video_cru":
                # Quadro sem espelhar, em até 960 px de largura, para as mãos 3D (demo/hand3d.html e demo/carro3d.html).
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=quadro")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                with estado.trava:
                    estado.quer_cru += 1
                try:
                    enviado = None
                    while not parar.is_set():
                        jpeg = estado.jpeg_cru
                        if jpeg and jpeg is not enviado:
                            self.wfile.write(b"--quadro\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                            self.wfile.flush()
                            enviado = jpeg
                        time.sleep(0.02)
                except OSError:
                    pass
                finally:
                    with estado.trava:
                        estado.quer_cru -= 1
            elif url.path == "/testar" and args.simular:
                comando = parse_qs(url.query).get("c", ["proxima"])[0]
                estado.publicar({"tipo": "comando", "comando": comando, "classe": comando, "gesto": "Teste", "confianca": 1.0,
                                 "repeticao": "repeticao" in parse_qs(url.query)})
                self.send_response(204)
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()

    return ThreadingHTTPServer(("127.0.0.1", args.porta), Atendente)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--camera", type=int, default=0)
    a.add_argument("--modelo", default=None)
    a.add_argument("--limiar", type=float, default=0.7, help="Confiança mínima para aceitar um comando")
    a.add_argument("--seguidas", type=int, default=None,
                   help="Janelas seguidas exigidas para disparar (padrão: 3 com NVIDIA, 2 sem)")
    a.add_argument("--pausa", type=float, default=1.2, help="Segundos sem novo comando depois de um disparo")
    a.add_argument("--suavizar", type=int, default=2)
    a.add_argument("--intervalo", type=float, default=0.15)
    a.add_argument("--janelas", type=float, nargs="+", default=None,
                   help="Durações das janelas, em segundos (padrão: 1 1,5 3 com NVIDIA, 1 1,5 sem)")
    a.add_argument("--porta", type=int, default=8765)
    a.add_argument("--pagina", default="player.html", help="Página de demo/ a servir")
    a.add_argument("--cpu", action="store_true")
    a.add_argument("--simular", action="store_true", help="Teste sem câmera, com ruído e a rota /testar")
    a.add_argument("--nao-abrir", action="store_true", help="Não abre o navegador")
    a.add_argument("--musicas", default=str(RAIZ / "musicas"), help="Pasta com os áudios servidos em /musicas/")
    a.add_argument("--repetir", type=float, default=2.0, help="Segundos entre repetições de um polegar mantido, no painel")
    a.add_argument("--painel", action="store_true", help="Liga os gestos extras do painel: polegares, tamborilar, puxar e empurrar a mão")
    a.add_argument("--hospital", action="store_true", help="Liga os gestos extras do hospital: polegares, acenar e zoom com a mão inteira")
    a.add_argument("--tela-cheia", action="store_true", help="Abre a página no Chrome ou no Edge em tela cheia")
    a.add_argument("--apresentacao", action="store_true", help="Dispara os gestos de todas as demos; a página repassa cada um ao slide aberto")
    a.add_argument("--carro3d", action="store_true", help="Comandos do carro 3D: os números (1 a 4 dedos) abrem uma página da central")
    a.add_argument("--pontos", default=None, help="GRU sobre os pontos da mão (resultados/pontos/modelo_final/gru.pt), combinada com a CNN")
    args = a.parse_args()
    if args.apresentacao:
        COMANDOS.clear()
        COMANDOS.update({g: g for g in GESTOS_APRESENTACAO})
    if args.painel:
        COMANDOS.update(COMANDOS_PAINEL)
    if args.hospital:
        COMANDOS.update(COMANDOS_HOSPITAL)
    if args.carro3d:
        COMANDOS.clear()
        COMANDOS.update(COMANDOS_CARRO3D)

    estado, parar = Estado(), threading.Event()
    servidor = criar_servidor(args, estado, parar)
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    endereco = f"http://127.0.0.1:{args.porta}/"
    print(f"Player em {endereco} (Ctrl+C encerra)", flush=True)
    if not args.nao_abrir:
        abrir_tela_cheia(endereco) if args.tela_cheia else webbrowser.open(endereco)
    try:
        laco_modelo(args, estado, parar)
        while not parar.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        parar.set()
        servidor.shutdown()


if __name__ == "__main__":
    main()
