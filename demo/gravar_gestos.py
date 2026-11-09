"""Grava vídeos de webcam para testar os gestos novos, em data/webcam/<pessoa>/.

Esses vídeos não entram no treino: são a medida de como o modelo funciona fora do dataset. Mostram rostos,
então ficam em data/, que não vai para o git.

Uso:
    python demo/gravar_gestos.py --pessoa ana
    (Q ou Esc interrompe; o que já foi gravado fica salvo)
"""
import argparse
import re
import time
from pathlib import Path

import cv2

GESTOS = [("one_finger", "UM DEDO", "Indicador levantado, palma para a camera"),
          ("two_fingers", "DOIS DEDOS", "Indicador e medio levantados"),
          ("three_fingers", "TRES DEDOS", "Indicador, medio e anelar levantados"),
          ("four_fingers", "QUATRO DEDOS", "Quatro dedos levantados, polegar dobrado"),
          ("fist", "PUNHO", "Mao fechada, parada"),
          ("stop_sign", "SINAL DE PARE", "Mao aberta, parada"),
          ("thumb_up", "POLEGAR PARA CIMA", "Parado"),
          ("thumb_down", "POLEGAR PARA BAIXO", "Parado"),
          ("sem_gesto", "SEM GESTO", "Aja normalmente: fale, olhe em volta, mexa as maos sem fazer gestos")]
QUADROS_POR_SEGUNDO = 12  # o mesmo ritmo do modelo


def texto(tela, linha, y, escala=0.9, cor=(255, 255, 255)):
    cv2.putText(tela, linha, (30, y), cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(tela, linha, (30, y), cv2.FONT_HERSHEY_SIMPLEX, escala, cor, 2, cv2.LINE_AA)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--pessoa", required=True, help="Só letras, números, - e _")
    a.add_argument("--camera", type=int, default=0)
    a.add_argument("--destino", type=Path, default=Path("data/webcam"))
    a.add_argument("--rodadas", type=int, default=2, help="Quantas vezes gravar cada gesto")
    a.add_argument("--segundos", type=float, default=8.0)
    a.add_argument("--segundos-sem-gesto", type=float, default=45.0)
    a.add_argument("--preparo", type=float, default=4.0, help="Segundos para posicionar a mão")
    args = a.parse_args()
    if not re.fullmatch(r"[\w-]+", args.pessoa):
        raise SystemExit("Use só letras, números, - e _ em --pessoa.")
    pasta = args.destino / args.pessoa
    pasta.mkdir(parents=True, exist_ok=True)

    captura = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not captura.isOpened():
        raise SystemExit(f"Não consegui abrir a câmera {args.camera}. Feche outros apps que a usem.")
    janela = "Gravar gestos"
    cv2.namedWindow(janela)

    def ler():
        ok, quadro = captura.read()
        if not ok:
            raise SystemExit("A câmera parou de enviar imagens.")
        return quadro

    def saiu():
        return cv2.waitKey(1) & 0xFF in (ord("q"), 27)

    gravados = 0
    try:
        for rodada in range(1, args.rodadas + 1):
            for nome, titulo, dica in GESTOS:
                fim = time.time() + args.preparo
                while (restante := fim - time.time()) > 0:
                    tela = cv2.flip(ler(), 1)
                    texto(tela, f"Prepare: {titulo}", 50, 1.2, (0, 220, 255))
                    texto(tela, dica, 95, 0.75)
                    texto(tela, f"Comeca em {restante:.0f} s (rodada {rodada}/{args.rodadas})", 140, 0.75)
                    cv2.imshow(janela, tela)
                    if saiu():
                        return
                quadro = ler()
                altura, largura = quadro.shape[:2]
                arquivo = pasta / f"{nome}_{rodada}.mp4"
                temporario = pasta / f"{nome}_{rodada}.tmp.mp4"
                saida = cv2.VideoWriter(str(temporario), cv2.VideoWriter_fourcc(*"mp4v"), QUADROS_POR_SEGUNDO, (largura, altura))
                duracao = args.segundos_sem_gesto if nome == "sem_gesto" else args.segundos
                inicio, proximo = time.time(), 0.0
                completo = True
                while (passado := time.time() - inicio) < duracao:
                    quadro = ler()
                    if passado >= proximo:
                        saida.write(quadro)
                        proximo += 1 / QUADROS_POR_SEGUNDO
                    tela = cv2.flip(quadro, 1)
                    texto(tela, f"GRAVANDO: {titulo}", 50, 1.2, (60, 60, 255))
                    texto(tela, dica, 95, 0.75)
                    texto(tela, f"{duracao - passado:.0f} s", 140, 0.75)
                    cv2.imshow(janela, tela)
                    if saiu():
                        completo = False
                        break
                saida.release()
                if not completo:
                    temporario.unlink(missing_ok=True)  # vídeo pela metade atrapalharia a medida
                    return
                temporario.replace(arquivo)
                gravados += 1
    finally:
        captura.release()
        cv2.destroyAllWindows()
        print(f"{gravados} vídeos gravados em {pasta}. Para medir: python src/avaliar_webcam.py", flush=True)


if __name__ == "__main__":
    main()
