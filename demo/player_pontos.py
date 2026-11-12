"""Versão do player.py com os números decididos pelos pontos da mão (MediaPipe) e a CNN 3D só do Jester.

As páginas (player, painel do carro, hospital, apresentação) são as mesmas e não mudam: o servidor,
os eventos e os nomes dos comandos vêm do player.py, importado sem alteração. Muda só quem decide:
  - um a quatro dedos e punho: os pontos da mão (src/pontos_mao.py);
  - todos os outros gestos: a CNN 3D (por padrão resultados/jester/cnn_3d/melhor.pt), com o mesmo gatilho.
O vídeo mostrado na página ganha o esqueleto da mão e a pose contada.

Uso (no ambiente com o MediaPipe, ver PONTOS_MAO.md):
    demo\\python_pontos.bat demo/player_pontos.py --pagina painel_carro.html --porta 8770 --painel
"""
import argparse
import collections
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import motor_pontos as MP
import player as PL

V, P = MP.V, MP.P


def laco(args, estado, parar):
    gatilho = None  # criado depois de saber o dispositivo, como no player.py

    def disparar_cnn(nome, confianca, agora, presente):
        return gatilho.atualizar(nome, confianca, agora, presente)

    sistema = MP.SistemaPontos(disparar_cnn, args.modelo, args.janelas, args.intervalo, args.suavizar, args.cpu,
                               args.tempo_estavel, args.movimento, args.escala, args.bloqueio, args.limiar)
    if args.seguidas is None:
        args.seguidas = 3 if sistema.cnn.dispositivo == "cuda" else 2
    gatilho = PL.Gatilho(args.limiar, args.seguidas, args.pausa,
                         PL.COMANDOS_VOLUME if (args.painel or args.apresentacao) else (), args.repetir)
    print(f"CNN: {Path(args.modelo).parent.name} (validação {sistema.cnn.info['top1_val']:.1%}), {sistema.cnn.dispositivo}; "
          f"números pelos pontos da mão", flush=True)

    captura = None
    if not args.simular:
        captura = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        captura.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not captura.isOpened():
            print(f"ERRO: não consegui abrir a câmera {args.camera}. Feche outros apps que a usem.", flush=True)
            estado.publicar({"tipo": "erro", "mensagem": "Não consegui abrir a câmera. Feche outros apps que a usem."})
            sistema.fechar()
            return

    tempos = collections.deque(maxlen=30)
    ultimo_video, ultimo_estado = 0.0, 0.0
    try:
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
            evento = sistema.processar(quadro, agora)
            disparo = evento["disparo"]
            if disparo and disparo in PL.COMANDOS:
                estado.disparos += 1
                repeticao = gatilho.foi_repeticao if evento["origem"] == "cnn" else False
                estado.publicar({"tipo": "comando", "comando": PL.COMANDOS[disparo], "classe": disparo,
                                 "gesto": V.TRADUCAO[disparo], "confianca": 1.0 if evento["origem"] == "pontos" else sistema.cnn_conf,
                                 "repeticao": repeticao, "origem": evento["origem"]})
            if disparo:
                print(f"{time.strftime('%H:%M:%S')} disparo ({evento['origem']}): {V.TRADUCAO.get(disparo, disparo)}"
                      + ("" if disparo in PL.COMANDOS else " (não é comando desta página)"), flush=True)
            if evento["suprimido"]:
                print(f"{time.strftime('%H:%M:%S')} CNN barrada (mão parada num número): {V.TRADUCAO.get(evento['suprimido'])}", flush=True)

            if agora - ultimo_video >= 0.08:
                tela = quadro.copy()
                if sistema.decisor.mao is not None:
                    P.desenhar_mao(tela, sistema.decisor.mao, dedos=sistema.decisor.dedos)
                pequena = cv2.resize(cv2.flip(tela, 1), (480, 270), interpolation=cv2.INTER_AREA)
                pose = sistema.decisor.pose
                if pose in P.NUMEROS:
                    cv2.putText(pequena, f"{V.TRADUCAO[pose]}  {sistema.decisor.progresso:.0%}", (10, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 205, 60), 2, cv2.LINE_AA)
                ok, codificado = cv2.imencode(".jpg", pequena, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    estado.jpeg = codificado.tobytes()
                ultimo_video = agora
            if agora - ultimo_estado >= 0.2:
                fps = (len(tempos) - 1) / max(tempos[-1] - tempos[0], 1e-3)
                minutos = max((agora - estado.inicio) / 60, 1e-3)
                nome, confianca = sistema.gesto_atual()
                estado.publicar({"tipo": "estado", "gesto": V.TRADUCAO.get(nome, nome), "e_comando": nome in PL.COMANDOS,
                                 "confianca": confianca, "disparos": estado.disparos, "por_minuto": estado.disparos / minutos,
                                 "fps": fps, "ms": sistema.ms, "ms_mao": sistema.ms_mao, "aquecendo": sistema.cnn.aquecendo,
                                 "pose": sistema.decisor.pose, "dedos": list(sistema.decisor.dedos or [])})
                ultimo_estado = agora
    finally:
        sistema.fechar()
        if captura is not None:
            captura.release()


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--camera", type=int, default=0)
    a.add_argument("--modelo", default=str(MP.MODELO_PADRAO), help="CNN dos gestos de movimento (padrão: cnn_3d, só Jester)")
    a.add_argument("--limiar", type=float, default=0.7, help="Confiança mínima para aceitar um comando da CNN")
    a.add_argument("--seguidas", type=int, default=None, help="Janelas seguidas exigidas para a CNN disparar (padrão: 3 com NVIDIA, 2 sem)")
    a.add_argument("--pausa", type=float, default=1.2, help="Segundos sem novo comando da CNN depois de um disparo")
    a.add_argument("--suavizar", type=int, default=2)
    a.add_argument("--intervalo", type=float, default=0.15)
    a.add_argument("--janelas", type=float, nargs="+", default=None)
    a.add_argument("--porta", type=int, default=8765)
    a.add_argument("--pagina", default="player.html", help="Página de demo/ a servir")
    a.add_argument("--cpu", action="store_true")
    a.add_argument("--simular", action="store_true", help="Teste sem câmera, com ruído e a rota /testar")
    a.add_argument("--nao-abrir", action="store_true", help="Não abre o navegador")
    a.add_argument("--musicas", default=str(PL.RAIZ / "musicas"))
    a.add_argument("--repetir", type=float, default=2.0)
    a.add_argument("--painel", action="store_true")
    a.add_argument("--hospital", action="store_true")
    a.add_argument("--tela-cheia", action="store_true")
    a.add_argument("--apresentacao", action="store_true")
    # Parte nova: regra dos números.
    a.add_argument("--tempo-estavel", type=float, default=0.5, help="Segundos com a mesma pose para um número disparar")
    a.add_argument("--movimento", type=float, default=0.5, help="Quanto o pulso pode andar, em tamanhos de palma, e ainda contar como parado")
    a.add_argument("--escala", type=float, default=0.25, help="Quanto a mão pode aproximar ou afastar (log do tamanho) e ainda contar como parada")
    a.add_argument("--bloqueio", type=float, default=1.0, help="Segundos em que um lado fica bloqueado depois de o outro disparar")
    args = a.parse_args()
    if args.apresentacao:
        PL.COMANDOS.clear()
        PL.COMANDOS.update({g: g for g in PL.GESTOS_APRESENTACAO})
    if args.painel:
        PL.COMANDOS.update(PL.COMANDOS_PAINEL)
    if args.hospital:
        PL.COMANDOS.update(PL.COMANDOS_HOSPITAL)

    estado, parar = PL.Estado(), threading.Event()
    servidor = PL.criar_servidor(args, estado, parar)
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    endereco = f"http://127.0.0.1:{args.porta}/"
    print(f"Player (CNN + pontos da mão) em {endereco} (Ctrl+C encerra)", flush=True)
    if not args.nao_abrir:
        PL.abrir_tela_cheia(endereco) if args.tela_cheia else PL.webbrowser.open(endereco)
    try:
        laco(args, estado, parar)
        while not parar.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        parar.set()
        servidor.shutdown()


if __name__ == "__main__":
    main()
