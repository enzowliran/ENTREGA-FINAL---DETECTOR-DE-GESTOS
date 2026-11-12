"""Janela da câmera com a parte nova: esqueleto da mão, dedos contados e o gesto disparado.

Como o ao_vivo.py, mas os números (um a quatro dedos e punho) vêm dos pontos da mão e os outros
gestos vêm da CNN 3D só do Jester. Os disparos seguem a mesma regra do player (gatilho com janelas
seguidas para a CNN; pose estável e mão parada para os números).

Teclas: Q ou Esc fecha, Z zera os contadores.

Uso (no ambiente com o MediaPipe, ver PONTOS_MAO.md):
    demo\\python_pontos.bat demo/ao_vivo_pontos.py
    demo\\python_pontos.bat demo/ao_vivo_pontos.py --tempo-estavel 0.7 --camera 1
"""
import argparse
import collections
import time

import cv2
import numpy as np

import motor_pontos as MP
import player as PL

V, P = MP.V, MP.P


def painel_dedos(tela, sistema, fonte_cv=cv2.FONT_HERSHEY_SIMPLEX):
    """Quadro no canto inferior direito: dedos esticados, pose e progresso até disparar."""
    d = sistema.decisor
    altura, largura = tela.shape[:2]
    x0, y0 = largura - 330, altura - 250
    cv2.rectangle(tela, (x0, y0), (largura - 10, altura - 106), (0, 0, 0), -1)
    if d.mao is None:
        cv2.putText(tela, "Pontos: nenhuma mao", (x0 + 10, y0 + 30), fonte_cv, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    else:
        marcas = "".join("I" if e else "." for e in d.dedos)
        cv2.putText(tela, f"Dedos (pol..min): {marcas}", (x0 + 10, y0 + 28), fonte_cv, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
        pose = V.TRADUCAO.get(d.pose, "nao e numero") if d.pose else "nao e numero"
        cv2.putText(tela, f"Pose: {pose}", (x0 + 10, y0 + 56), fonte_cv, 0.6, (60, 205, 255), 1, cv2.LINE_AA)
        cv2.putText(tela, "parada" if d.parado else "em movimento", (x0 + 10, y0 + 84), fonte_cv, 0.6,
                    (120, 230, 150) if d.parado else (90, 160, 255), 1, cv2.LINE_AA)
        cv2.rectangle(tela, (x0 + 10, y0 + 100), (largura - 20, y0 + 112), (70, 70, 70), -1)
        cv2.rectangle(tela, (x0 + 10, y0 + 100), (x0 + 10 + int((largura - 30 - x0) * d.progresso), y0 + 112), (60, 205, 255), -1)
    supr = sum(sistema.suprimidos.values())
    cv2.putText(tela, f"Mao: {sistema.ms_mao:.0f} ms  CNN barrada: {supr}", (x0 + 10, y0 + 136), fonte_cv, 0.5,
                (255, 205, 120), 1, cv2.LINE_AA)
    return tela


def rodar(args):
    PL.COMANDOS.clear()  # nesta janela, qualquer gesto da CNN pode disparar

    def disparar_cnn(nome, confianca, agora, presente):
        return gatilho.atualizar(nome, confianca, agora, presente)

    sistema = MP.SistemaPontos(disparar_cnn, args.modelo, args.janelas, args.intervalo, args.suavizar, args.cpu,
                               args.tempo_estavel, args.movimento, args.escala, args.bloqueio, args.limiar)
    PL.COMANDOS.update({n: n for n in sistema.cnn.nomes if n not in V.SEM_GESTO and n not in P.NUMEROS})
    seguidas = args.seguidas or (3 if sistema.cnn.dispositivo == "cuda" else 2)
    gatilho = PL.Gatilho(args.limiar, seguidas, args.pausa)
    print(f"CNN: {args.modelo} ({sistema.cnn.dispositivo}); números pelos pontos da mão", flush=True)

    captura = None
    if not args.simular:
        captura = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        captura.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not captura.isOpened():
            sistema.fechar()
            raise SystemExit(f"Não consegui abrir a câmera {args.camera}. Feche outros apps que a usem ou tente --camera 1.")
        cv2.namedWindow("Gestos ao vivo (pontos)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gestos ao vivo (pontos)", 1100, 620)

    fontes = (V.fonte(38), V.fonte(17))
    tempos = collections.deque(maxlen=30)
    legenda, destaque, fim_legenda, disparos = "Aquecendo...", False, 0.0, 0
    inicio, simulados = time.time(), 0
    try:
        while True:
            if args.simular:
                quadro = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
                simulados += 1
                time.sleep(0.03)
            else:
                ok, quadro = captura.read()
                if not ok:
                    print("A câmera parou de enviar imagens.")
                    break
            agora = time.time()
            tempos.append(agora)
            fps = (len(tempos) - 1) / (tempos[-1] - tempos[0]) if len(tempos) > 1 and tempos[-1] > tempos[0] else 0.0
            evento = sistema.processar(quadro, agora)
            if evento["disparo"]:
                disparos += 1
                origem = "pontos" if evento["origem"] == "pontos" else "CNN"
                legenda = f"{V.TRADUCAO.get(evento['disparo'], evento['disparo'])} ({origem})"
                destaque, fim_legenda = True, agora + args.segurar
                print(f"{time.strftime('%H:%M:%S')} disparo ({evento['origem']}): {V.TRADUCAO.get(evento['disparo'])}", flush=True)
            elif agora >= fim_legenda:
                nome, _ = sistema.gesto_atual()
                destaque = False
                legenda = ("Aquecendo..." if sistema.cnn.aquecendo else
                           "Sem gesto" if nome in V.SEM_GESTO else V.TRADUCAO.get(nome, nome))

            tela = quadro.copy()
            if sistema.decisor.mao is not None:
                P.desenhar_mao(tela, sistema.decisor.mao, dedos=sistema.decisor.dedos)
            tela = cv2.flip(tela, 1) if not args.nao_espelhar else tela
            top3 = []
            if sistema.cnn_media is not None:
                ordem = sistema.cnn_media.argsort()[::-1][:3]
                top3 = [("CNN: " + V.TRADUCAO.get(sistema.cnn.nomes[i], sistema.cnn.nomes[i]), float(sistema.cnn_media[i])) for i in ordem]
            _, confianca = sistema.gesto_atual()
            tela = V.desenhar(tela, legenda, confianca, destaque, top3, disparos, (agora - inicio) / 60, sistema.ms,
                              sistema.cnn.aquecendo, fontes, fps=fps)
            tela = painel_dedos(tela, sistema)
            if args.simular:
                if simulados >= args.simular:
                    cv2.imwrite(args.saida, tela)
                    print(f"Simulação: {simulados} quadros, CNN {sistema.ms:.0f} ms, mão {sistema.ms_mao:.0f} ms; imagem em {args.saida}")
                    break
                continue
            cv2.imshow("Gestos ao vivo (pontos)", tela)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty("Gestos ao vivo (pontos)", cv2.WND_PROP_VISIBLE) < 1:
                break
            if tecla in (ord("z"), ord("Z")):
                disparos, inicio = 0, time.time()
                sistema.suprimidos.clear()
    finally:
        sistema.fechar()
        if captura is not None:
            captura.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--camera", type=int, default=0)
    a.add_argument("--modelo", default=str(MP.MODELO_PADRAO), help="CNN dos gestos de movimento (padrão: cnn_3d, só Jester)")
    a.add_argument("--limiar", type=float, default=0.7)
    a.add_argument("--seguidas", type=int, default=None, help="Padrão: 3 com NVIDIA, 2 sem")
    a.add_argument("--pausa", type=float, default=1.2)
    a.add_argument("--suavizar", type=int, default=2)
    a.add_argument("--intervalo", type=float, default=0.15)
    a.add_argument("--janelas", type=float, nargs="+", default=None)
    a.add_argument("--segurar", type=float, default=1.5, help="Segundos que a legenda do disparo permanece")
    a.add_argument("--tempo-estavel", type=float, default=0.5)
    a.add_argument("--movimento", type=float, default=0.5)
    a.add_argument("--escala", type=float, default=0.25)
    a.add_argument("--bloqueio", type=float, default=1.0)
    a.add_argument("--nao-espelhar", action="store_true")
    a.add_argument("--cpu", action="store_true")
    a.add_argument("--simular", type=int, default=0, help="Teste sem câmera: número de quadros de ruído")
    a.add_argument("--saida", default="teste_demo_pontos.png")
    rodar(a.parse_args())
