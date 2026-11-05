"""Demonstração ao vivo: abre a webcam e escreve o gesto detectado como legenda na tela.

Usa o modelo treinado no Jester. A tecla Q (ou Esc) fecha, e Z zera o contador de disparos.

Uso:
    python demo/ao_vivo.py
    python demo/ao_vivo.py --camera 1 --limiar 0.7 --modelo resultados/jester/cnn_3d/melhor.pt
"""
import argparse
import collections
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import modelos as M

RAIZ = Path(__file__).resolve().parent.parent
MODELOS_PADRAO = [RAIZ / "resultados/jester/cnn_3d/melhor.pt", RAIZ / "resultados/jester/cnn_3d_medio/melhor.pt",
                  RAIZ / "resultados/jester/cnn_3d_pequeno/melhor.pt"]
N_FRAMES, TAMANHO = 16, 96
# O buffer guarda 12 frames por segundo, como o Jester. Cada janela sorteia 16 deles, então uma
# janela curta enxerga gestos rápidos e uma longa enxerga gestos lentos.
QUADROS_POR_SEGUNDO = 12
SEM_GESTO = {"No gesture", "Doing other things"}
TRADUCAO = {
    "Swiping Left": "Deslizar para a esquerda", "Swiping Right": "Deslizar para a direita",
    "Swiping Down": "Deslizar para baixo", "Swiping Up": "Deslizar para cima",
    "Pushing Hand Away": "Empurrar a mão", "Pulling Hand In": "Puxar a mão",
    "Sliding Two Fingers Left": "Dois dedos para a esquerda", "Sliding Two Fingers Right": "Dois dedos para a direita",
    "Sliding Two Fingers Down": "Dois dedos para baixo", "Sliding Two Fingers Up": "Dois dedos para cima",
    "Pushing Two Fingers Away": "Empurrar com dois dedos", "Pulling Two Fingers In": "Puxar com dois dedos",
    "Rolling Hand Forward": "Rolar a mão para frente", "Rolling Hand Backward": "Rolar a mão para trás",
    "Turning Hand Clockwise": "Girar a mão no sentido horário", "Turning Hand Counterclockwise": "Girar a mão no sentido anti-horário",
    "Zooming In With Full Hand": "Zoom in com a mão", "Zooming Out With Full Hand": "Zoom out com a mão",
    "Zooming In With Two Fingers": "Zoom in com dois dedos", "Zooming Out With Two Fingers": "Zoom out com dois dedos",
    "Thumb Up": "Polegar para cima", "Thumb Down": "Polegar para baixo", "Shaking Hand": "Acenar",
    "Stop Sign": "Sinal de pare", "Drumming Fingers": "Tamborilar os dedos",
    "No gesture": "Sem gesto", "Doing other things": "Outro movimento",
}


def carregar_modelo(caminho, dispositivo):
    estado = torch.load(caminho, map_location=dispositivo, weights_only=False)
    modelo = M.criar_modelo(estado["arquitetura"], len(estado["nomes"]))
    modelo.load_state_dict(estado["modelo"])
    return modelo.to(dispositivo).eval(), estado["nomes"], estado


def preparar_quadro(quadro):
    """Mesmo tratamento da validação: recorte central quadrado, 96x96 e RGB."""
    altura, largura = quadro.shape[:2]
    lado = min(altura, largura)
    y0, x0 = (altura - lado) // 2, (largura - lado) // 2
    recorte = cv2.resize(quadro[y0:y0 + lado, x0:x0 + lado], (TAMANHO, TAMANHO), interpolation=cv2.INTER_LINEAR)
    return recorte[..., ::-1].transpose(2, 0, 1).astype(np.float32)


def fonte(tamanho):
    for nome in ("segoeuib.ttf", "arialbd.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(nome, tamanho)
        except OSError:
            continue
    return ImageFont.load_default()


def desenhar(quadro, legenda, confianca, destaque, top3, disparos, minutos, ms_inferencia, aquecendo, fontes,
             entrada=None, fps=0.0, janela_ativa=None):
    imagem = Image.fromarray(quadro[..., ::-1]).convert("RGBA")
    camada = Image.new("RGBA", imagem.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(camada)
    largura, altura = imagem.size
    d.rectangle([0, altura - 96, largura, altura], fill=(0, 0, 0, 200))
    cor = (120, 230, 150, 255) if destaque else (205, 205, 200, 255)
    d.text((largura / 2, altura - 62), legenda, font=fontes[0], fill=cor, anchor="mm")
    if not aquecendo:
        d.rectangle([largura * 0.2, altura - 24, largura * 0.8, altura - 16], fill=(70, 70, 70, 255))
        d.rectangle([largura * 0.2, altura - 24, largura * (0.2 + 0.6 * confianca), altura - 16], fill=cor)
    d.rectangle([0, 0, 370, 112], fill=(0, 0, 0, 150))
    for i, (nome, p) in enumerate(top3):
        d.text((10, 8 + i * 24), f"{p * 100:3.0f}%  {nome}", font=fontes[1], fill=(235, 235, 230, 255))
    taxa = disparos / minutos if minutos > 0.05 else 0.0
    d.text((10, 84), f"Disparos: {disparos} ({taxa:.1f}/min) · {ms_inferencia:.0f} ms", font=fontes[1], fill=(255, 205, 120, 255))
    janela_texto = f" · janela {janela_ativa:.1f} s" if janela_ativa else ""
    d.text((largura - 10, 8), f"Câmera: {fps:.0f} fps{janela_texto}", font=fontes[1], fill=(235, 235, 230, 255), anchor="ra")
    final = Image.alpha_composite(imagem, camada).convert("RGB")
    if entrada is not None:
        # Miniatura com o que o modelo recebe, para conferir o tamanho da mão na imagem.
        mini = Image.fromarray(entrada.transpose(1, 2, 0).astype(np.uint8)).resize((192, 192), Image.NEAREST)
        final.paste(mini, (largura - 202, 34))
    return np.array(final)[..., ::-1].copy()


def rodar(args):
    dispositivo = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    caminho = Path(args.modelo) if args.modelo else next((p for p in MODELOS_PADRAO if p.exists()), None)
    if caminho is None or not caminho.exists():
        raise SystemExit("Nenhum modelo treinado encontrado em resultados/jester.")
    modelo, nomes, estado = carregar_modelo(caminho, dispositivo)
    print(f"Modelo: {caminho} (época {estado['epoca'] + 1}, validação {estado['top1_val']:.1%}), dispositivo {dispositivo}")

    fontes = (fonte(38), fonte(17))
    captura = None
    if not args.simular:
        captura = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        captura.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not captura.isOpened():
            raise SystemExit(f"Não consegui abrir a câmera {args.camera}. Feche outros apps que a usem ou tente --camera 1.")
        cv2.namedWindow("Gestos ao vivo", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Gestos ao vivo", 1100, 620)

    maximo = int(round(max(args.janelas) * QUADROS_POR_SEGUNDO))
    janela = collections.deque(maxlen=maximo)
    minimo = int(round(min(args.janelas) * QUADROS_POR_SEGUNDO))
    tempos = collections.deque(maxlen=30)
    janela_ativa = None
    historico = collections.deque(maxlen=args.suavizar)
    ultimo_quadro, ultima_inferencia = 0.0, 0.0
    legenda, confianca, destaque = "Aquecendo...", 0.0, False
    ativo, fim_legenda, disparos, top3, ms = None, 0.0, 0, [], 0.0
    inicio = time.time()
    simulados = 0

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
        # A entrada do modelo não é espelhada, como no treino. Só a imagem exibida é espelhada.
        if agora - ultimo_quadro >= 1 / QUADROS_POR_SEGUNDO:
            janela.append(preparar_quadro(quadro))
            ultimo_quadro = agora
        if len(janela) >= minimo and agora - ultima_inferencia >= args.intervalo:
            t0 = time.perf_counter()
            buffer = list(janela)
            lotes_janela, usadas = [], []
            for duracao in args.janelas:
                n = int(round(duracao * QUADROS_POR_SEGUNDO))
                if len(buffer) >= n:
                    indices = np.round(np.linspace(len(buffer) - n, len(buffer) - 1, N_FRAMES)).astype(int)
                    lotes_janela.append(np.stack([buffer[i] for i in indices]))
                    usadas.append(duracao)
            with torch.no_grad(), torch.autocast(dispositivo, dtype=torch.bfloat16):
                x = (torch.from_numpy(np.stack(lotes_janela)).to(dispositivo) - 127.5) / 127.5
                todas = modelo(x).float().softmax(1).cpu().numpy()
            # Fica a janela com o gesto mais confiante. Isso favorece detectar, então pode aumentar os falsos disparos.
            indices_gesto = [i for i, nome in enumerate(nomes) if nome not in SEM_GESTO]
            escolhida = int(np.argmax([q[indices_gesto].max() for q in todas]))
            p, janela_ativa = todas[escolhida], usadas[escolhida]
            ms = (time.perf_counter() - t0) * 1000
            ultima_inferencia = agora
            historico.append(p)
            media = np.mean(historico, axis=0)
            ordem = media.argsort()[::-1][:3]
            top3 = [(TRADUCAO.get(nomes[i], nomes[i]), float(media[i])) for i in ordem]
            melhor = nomes[ordem[0]]
            confianca = float(media[ordem[0]])
            if melhor not in SEM_GESTO and confianca >= args.limiar:
                if melhor != ativo:
                    disparos += 1
                    ativo = melhor
                legenda, destaque, fim_legenda = TRADUCAO.get(melhor, melhor), True, agora + args.segurar
            elif agora >= fim_legenda:
                ativo, destaque = None, False
                legenda = "Sem gesto" if melhor in SEM_GESTO else "Incerto"
        elif len(janela) < minimo:
            legenda = f"Aquecendo... {len(janela)}/{minimo}"

        tela = cv2.flip(quadro, 1) if not args.nao_espelhar else quadro
        tela = desenhar(tela, legenda, confianca, destaque, top3, disparos, (agora - inicio) / 60, ms, len(janela) < minimo, fontes,
                        entrada=janela[-1] if janela else None, fps=fps, janela_ativa=janela_ativa)
        if args.simular:
            if simulados >= args.simular:
                cv2.imwrite(args.saida, tela)
                print(f"Simulação: {simulados} quadros, inferência {ms:.1f} ms, top3 {top3[:1]}; imagem em {args.saida}")
                break
            continue
        cv2.imshow("Gestos ao vivo", tela)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty("Gestos ao vivo", cv2.WND_PROP_VISIBLE) < 1:
            break
        if tecla in (ord("z"), ord("Z")):
            disparos, inicio = 0, time.time()
    if captura is not None:
        captura.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--camera", type=int, default=0)
    a.add_argument("--modelo", default=None)
    a.add_argument("--limiar", type=float, default=0.5, help="Confiança mínima para mostrar um gesto")
    a.add_argument("--suavizar", type=int, default=2, help="Número de predições na média móvel")
    a.add_argument("--intervalo", type=float, default=0.15, help="Segundos entre inferências")
    a.add_argument("--janelas", type=float, nargs="+", default=[1.0, 1.5, 3.0], help="Durações das janelas, em segundos")
    a.add_argument("--segurar", type=float, default=1.5, help="Segundos que a legenda permanece")
    a.add_argument("--nao-espelhar", action="store_true", help="Não espelha a imagem exibida")
    a.add_argument("--cpu", action="store_true")
    a.add_argument("--simular", type=int, default=0, help="Teste sem câmera: número de quadros de ruído")
    a.add_argument("--saida", default="teste_demo.png")
    rodar(a.parse_args())
