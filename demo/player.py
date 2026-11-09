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
import sys
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
                     "Zooming In With Full Hand": "zoom_mais", "Zooming Out With Full Hand": "zoom_menos"}
TIPOS_AUDIO = {".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".wav": "audio/wav", ".ogg": "audio/ogg"}


class Estado:
    def __init__(self):
        self.trava = threading.Lock()
        self.jpeg = None
        self.assinantes = []
        self.inicio = time.time()
        self.disparos = 0

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
    print(f"Modelo: {caminho.parent.name} (validação {info['top1_val']:.1%}), {dispositivo}", flush=True)
    indices_gesto = [i for i, n in enumerate(nomes) if n not in V.SEM_GESTO]

    captura = None
    if not args.simular:
        captura = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        captura.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not captura.isOpened():
            print(f"ERRO: não consegui abrir a câmera {args.camera}. Feche outros apps que a usem.", flush=True)
            estado.publicar({"tipo": "erro", "mensagem": "Não consegui abrir a câmera. Feche outros apps que a usem."})
            return

    if dispositivo == "cpu":
        # Sem NVIDIA, câmera e modelo rodam separados. Com NVIDIA, o laço abaixo segue igual.
        laco_separado(args, estado, parar, captura, modelo, nomes, indices_gesto)
        return

    maximo = int(round(max(args.janelas) * V.QUADROS_POR_SEGUNDO))
    minimo = int(round(min(args.janelas) * V.QUADROS_POR_SEGUNDO))
    buffer, historico = collections.deque(maxlen=maximo), collections.deque(maxlen=args.suavizar)
    gatilho = Gatilho(args.limiar, args.seguidas, args.pausa, COMANDOS_VOLUME if args.painel else (), args.repetir)
    ultimo_quadro, ultima_inferencia, ultimo_video, ultimo_estado = 0.0, 0.0, 0.0, 0.0
    tempos = collections.deque(maxlen=30)
    nome_atual, confianca, ms = "Aquecendo", 0.0, 0.0

    while not parar.is_set():
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
                estado.publicar({"tipo": "comando", "comando": COMANDOS[disparo], "gesto": V.TRADUCAO[disparo],
                                 "confianca": confianca, "repeticao": gatilho.foi_repeticao})
        if agora - ultimo_video >= 0.08:
            pequena = cv2.resize(cv2.flip(quadro, 1), (480, 270), interpolation=cv2.INTER_AREA)
            ok, codificado = cv2.imencode(".jpg", pequena, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                estado.jpeg = codificado.tobytes()
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
        tempos = collections.deque(maxlen=30)
        ultimo_quadro, ultimo_video = 0.0, 0.0
        while not parar.is_set():
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
            if agora - ultimo_video >= 0.08:
                pequena = cv2.resize(cv2.flip(quadro, 1), (480, 270), interpolation=cv2.INTER_AREA)
                ok, codificado = cv2.imencode(".jpg", pequena, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    estado.jpeg = codificado.tobytes()
                ultimo_video = agora

    leitor = threading.Thread(target=ler_camera, daemon=True)
    leitor.start()

    historico = collections.deque(maxlen=args.suavizar)
    gatilho = Gatilho(args.limiar, args.seguidas, args.pausa, COMANDOS_VOLUME if args.painel else (), args.repetir)
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
                estado.publicar({"tipo": "comando", "comando": COMANDOS[disparo], "gesto": V.TRADUCAO[disparo],
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
            elif url.path == "/testar" and args.simular:
                comando = parse_qs(url.query).get("c", ["proxima"])[0]
                estado.publicar({"tipo": "comando", "comando": comando, "gesto": "Teste", "confianca": 1.0})
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
    args = a.parse_args()
    if args.painel:
        COMANDOS.update(COMANDOS_PAINEL)
    if args.hospital:
        COMANDOS.update(COMANDOS_HOSPITAL)

    estado, parar = Estado(), threading.Event()
    servidor = criar_servidor(args, estado, parar)
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    endereco = f"http://127.0.0.1:{args.porta}/"
    print(f"Player em {endereco} (Ctrl+C encerra)", flush=True)
    if not args.nao_abrir:
        webbrowser.open(endereco)
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
