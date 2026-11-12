"""Testador guiado na webcam para a parte nova (CNN 3D + pontos da mão), com o mesmo roteiro do testar_webcam.py.

Usa o testar_webcam.py sem alterá-lo: só troca o "modelo" dele pelo SistemaPontos (motor_pontos.py).
As tentativas, as teclas, o bloco "sem gesto" e os arquivos gravados são os mesmos. As sessões vão
para a mesma pasta resultados/webcam/, com o modelo anotado como "<cnn> + pontos", então o resumo
de sempre (python demo/testar_webcam.py --resumo) mostra a CNN sozinha e a parte nova lado a lado.

Diferença no cronômetro: aqui a CNN roda numa thread própria (como no player), então a câmera não
congela enquanto ela analisa. No testar_webcam.py original, sem NVIDIA, a análise é feita no mesmo laço.

Uso (no ambiente com o MediaPipe, ver PONTOS_MAO.md):
    demo\\python_pontos.bat demo/testar_webcam_pontos.py --pessoa enzo --conjunto dedos
"""
import json
import sys

import numpy as np

import motor_pontos as MP
import testar_webcam as T

P = MP.P
GatilhoOriginal = T.Gatilho
REGRA_PONTOS = {"tempo_estavel_s": 0.5, "movimento_palmas": 0.5, "escala_log": 0.25, "bloqueio_s": 1.0}
_atual = {}


class ModeloPontos:
    """Imita a interface de testar_webcam.Modelo, mas devolve os disparos já decididos pelo SistemaPontos."""

    def __init__(self, args):
        gatilho = GatilhoOriginal(args.limiar, args.seguidas, args.pausa)
        self.sistema = MP.SistemaPontos(lambda nome, conf, agora, presente: gatilho.atualizar(nome, conf, agora),
                                        args.modelo, args.janelas, args.intervalo, args.suavizar, False,
                                        REGRA_PONTOS["tempo_estavel_s"], REGRA_PONTOS["movimento_palmas"],
                                        REGRA_PONTOS["escala_log"], REGRA_PONTOS["bloqueio_s"], args.limiar)
        cnn = self.sistema.cnn
        self.nomes_cnn = list(cnn.nomes)
        self.nomes = self.nomes_cnn + [n for n in P.NUMEROS if n not in self.nomes_cnn]
        self.janelas, self.media, self.disparo = cnn.janelas, None, None
        _atual["modelo"] = self
        print(f"CNN: {args.modelo} (validação {cnn.info['top1_val']:.1%}), {cnn.dispositivo}; números pelos pontos da mão", flush=True)

    @property
    def buffer(self):
        return self.sistema.cnn.buffer

    @property
    def aquecendo(self):
        return self.sistema.cnn.aquecendo

    def receber(self, quadro, agora):
        evento = self.sistema.processar(quadro, agora)
        self.disparo = evento["disparo"]
        # "Confiança" de cada classe para as telas e para a coluna confianca_max_no_pedido:
        # a CNN para os gestos dela, e o progresso da pose (0 a 1) para os números.
        media = np.zeros(len(self.nomes), dtype=np.float32)
        if self.sistema.cnn_media is not None:
            media[:len(self.nomes_cnn)] = self.sistema.cnn_media
        for nome, valor in self.sistema.decisor.confiancas().items():
            media[self.nomes.index(nome)] = valor
        self.media = media
        nome, conf = self.sistema.gesto_atual()
        return nome, conf, media


class GatilhoPontos:
    """O disparo já vem decidido pelo SistemaPontos; aqui só é repassado ao testar_webcam."""

    def __init__(self, *_):
        pass

    def atualizar(self, nome, confianca, agora):
        return _atual["modelo"].disparo


def main():
    if "--resumo" not in sys.argv and "--modelo" not in sys.argv:
        sys.argv += ["--modelo", str(MP.MODELO_PADRAO)]
    T.Modelo, T.Gatilho = ModeloPontos, GatilhoPontos
    antes = {p.name for p in T.PASTA.iterdir()} if T.PASTA.exists() else set()
    try:
        T.main()
    finally:
        modelo = _atual.get("modelo")
        if modelo is not None:
            modelo.sistema.fechar()
    # Marca as sessões novas como da parte nova, para o resumo não misturar com a CNN sozinha.
    for pasta in (T.PASTA.iterdir() if T.PASTA.exists() else []):
        arquivo = pasta / "resumo.json"
        if pasta.name in antes or not arquivo.exists():
            continue
        r = json.loads(arquivo.read_text(encoding="utf-8"))
        if "+ pontos" not in r.get("modelo", ""):
            r["modelo"] = r.get("modelo", "?") + " + pontos"
            r["regra"]["pontos"] = REGRA_PONTOS
            r["regra"]["cnn_barrada_por_numero"] = dict(modelo.sistema.suprimidos) if modelo else {}
            arquivo.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"Sessão marcada como '{r['modelo']}' em {arquivo}")


if __name__ == "__main__":
    main()
