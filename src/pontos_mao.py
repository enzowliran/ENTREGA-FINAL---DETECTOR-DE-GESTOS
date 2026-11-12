"""Números feitos com a mão a partir dos 21 pontos do MediaPipe Hand Landmarker.

Parte nova do projeto: a CNN 3D continua com os gestos de movimento do Jester, e os números
(um a quatro dedos e punho) passam a ser decididos aqui, pela forma da mão.

Três peças:
  - DetectorMao: roda o Hand Landmarker (modelo do Google, baixado sozinho na primeira vez) e
    devolve os pontos da mão mais confiante do quadro.
  - contar_dedos / classificar_pose: diz quais dedos estão esticados, por regras, sem treino.
    Usa os pontos 3D em metros ("world landmarks"), que não dependem da distância da câmera nem
    de a mão estar inclinada. Conta qualquer combinação de dedos: indicador e médio é 2, polegar,
    indicador e médio é 3. Só o polegar sozinho não é número (é o joinha, da CNN), e cinco
    dedos também não (é a mão aberta, que fica com a CNN).
  - DecisorNumeros: decide quando um número dispara e quando a CNN pode disparar. Um número só
    vale com a mesma pose por `tempo_estavel` segundos e a mão parada; enquanto a mão está parada
    numa pose de número, a CNN não dispara; depois de qualquer disparo, o outro lado fica
    bloqueado por `bloqueio` segundos.

Precisa do ambiente separado com o MediaPipe: demo/python_pontos.bat (veja PONTOS_MAO.md).
"""
import collections
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
MODELO_MAO = RAIZ / "modelos_externos" / "hand_landmarker.task"
URL_MODELO_MAO = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
                  "float16/latest/hand_landmarker.task")

NUMEROS = ("One Finger", "Two Fingers", "Three Fingers", "Four Fingers", "Fist")
POR_CONTAGEM = {0: "Fist", 1: "One Finger", 2: "Two Fingers", 3: "Three Fingers", 4: "Four Fingers"}
NOMES_DEDOS = ("polegar", "indicador", "médio", "anelar", "mínimo")
# Pontos de cada dedo, da base para a ponta: (MCP, PIP, DIP, ponta). O polegar usa (CMC, MCP, IP, ponta).
DEDOS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
LIGACOES = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17))

# Limiares das regras. Os valores vêm de src/avaliar_pontos_hagrid.py, ajustados numa metade da
# amostra do HaGRID e medidos na outra (resultados/pontos/hagrid_amostra.json).
# Valores iniciais, escolhidos à mão antes da medição: reta 0,80, angulo 60, polegar_reta 0,85, polegar_fora 1,20.
LIMIARES = {
    "reta": 0.85,          # dedo esticado: ponta longe da base em relação ao comprimento do dedo (1 = reto)
    "angulo": 65.0,        # e apontando para fora da palma: ângulo entre pulso→base e base→ponta, em graus
    "polegar_reta": 0.95,  # polegar esticado: mesma medida de "reta", da base do MCP até a ponta
    "polegar_fora": 1.20,  # e afastado: distância da ponta ao mínimo, dividida pela largura da palma
}
LIMIARES_INICIAIS = {"reta": 0.80, "angulo": 60.0, "polegar_reta": 0.85, "polegar_fora": 1.20}


@dataclass
class Mao:
    imagem: np.ndarray   # 21 x 3: x e y de 0 a 1 na imagem, z relativo
    mundo: np.ndarray    # 21 x 3, em metros, com origem perto do centro da mão
    lado: str            # "Left" ou "Right", como o MediaPipe vê a imagem (não espelhada)
    confianca: float


def baixar_modelo_mao(caminho=MODELO_MAO):
    if not caminho.exists():
        caminho.parent.mkdir(parents=True, exist_ok=True)
        print(f"Baixando o Hand Landmarker do Google para {caminho} (cerca de 8 MB)...", flush=True)
        temporario = caminho.with_suffix(".parcial")
        urllib.request.urlretrieve(URL_MODELO_MAO, temporario)
        temporario.replace(caminho)
    return caminho


class DetectorMao:
    """Hand Landmarker do MediaPipe. Em modo vídeo, ele acompanha a mão entre quadros e fica mais rápido."""

    def __init__(self, video=True, maos=1, confianca=0.5, largura=640):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision

        self.mp, self.video, self.largura = mp, video, largura
        opcoes = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(baixar_modelo_mao())),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_hands=maos, min_hand_detection_confidence=confianca,
            min_hand_presence_confidence=confianca, min_tracking_confidence=confianca)
        self.detector = vision.HandLandmarker.create_from_options(opcoes)
        self.ultimo_ms = -1

    def detectar(self, quadro_bgr, agora=None):
        """Recebe um quadro BGR do OpenCV e devolve a Mao mais confiante, ou None."""
        import cv2

        altura, largura = quadro_bgr.shape[:2]
        if self.largura and largura > self.largura:
            quadro_bgr = cv2.resize(quadro_bgr, (self.largura, round(altura * self.largura / largura)), interpolation=cv2.INTER_AREA)
        rgb = np.ascontiguousarray(quadro_bgr[..., ::-1])
        imagem = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        if self.video:
            # O modo vídeo exige marcas de tempo sempre crescentes, em milissegundos.
            ms = int((agora if agora is not None else __import__("time").time()) * 1000)
            ms = max(ms, self.ultimo_ms + 1)
            self.ultimo_ms = ms
            r = self.detector.detect_for_video(imagem, ms)
        else:
            r = self.detector.detect(imagem)
        if not r.hand_landmarks:
            return None
        melhor = max(range(len(r.hand_landmarks)), key=lambda i: r.handedness[i][0].score)
        return Mao(imagem=np.array([[p.x, p.y, p.z] for p in r.hand_landmarks[melhor]], dtype=np.float32),
                   mundo=np.array([[p.x, p.y, p.z] for p in r.hand_world_landmarks[melhor]], dtype=np.float32),
                   lado=r.handedness[melhor][0].category_name, confianca=float(r.handedness[melhor][0].score))

    def fechar(self):
        self.detector.close()


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def _angulo(u, v):
    cos = float(np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-9))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def medidas(mundo):
    """Medidas de cada dedo usadas pelas regras. Ficam separadas para o avaliador poder ajustar os limiares."""
    p = mundo
    m = {}
    for nome, (a, b, c, d) in zip(NOMES_DEDOS[1:], DEDOS[1:]):
        comprimento = _dist(p[a], p[b]) + _dist(p[b], p[c]) + _dist(p[c], p[d])
        m[f"{nome}_reta"] = _dist(p[a], p[d]) / max(comprimento, 1e-9)
        m[f"{nome}_angulo"] = _angulo(p[a] - p[0], p[d] - p[a])
    m["polegar_reta"] = _dist(p[2], p[4]) / max(_dist(p[2], p[3]) + _dist(p[3], p[4]), 1e-9)
    # Ponta do polegar longe da base do mínimo = polegar para fora. Dobrado sobre a palma (punho, quatro), fica perto.
    m["polegar_fora"] = _dist(p[4], p[17]) / max(_dist(p[5], p[17]), 1e-9)
    return m


def contar_dedos(mundo, limiares=LIMIARES):
    """Devolve 5 booleanos, do polegar ao mínimo: True = dedo esticado."""
    m = medidas(mundo)
    polegar = m["polegar_reta"] >= limiares["polegar_reta"] and m["polegar_fora"] >= limiares["polegar_fora"]
    outros = [m[f"{n}_reta"] >= limiares["reta"] and m[f"{n}_angulo"] <= limiares["angulo"] for n in NOMES_DEDOS[1:]]
    return (polegar, *outros)


def classificar_pose(dedos):
    """Nome da classe do número, ou None quando a pose não é número (joinha, mão aberta)."""
    total = sum(dedos)
    if total == 1 and dedos[0]:
        return None  # só o polegar: joinha ou polegar para baixo, que são da CNN
    return POR_CONTAGEM.get(total)  # 5 dedos não está no dicionário: mão aberta fica com a CNN


def desenhar_mao(quadro_bgr, mao, espelhar=False, cor=(80, 220, 120), dedos=None):
    """Desenha o esqueleto da mão. Com `dedos`, as pontas esticadas ficam em destaque."""
    import cv2

    altura, largura = quadro_bgr.shape[:2]
    xy = mao.imagem[:, :2].copy()
    if espelhar:
        xy[:, 0] = 1 - xy[:, 0]
    pts = [(int(x * largura), int(y * altura)) for x, y in xy]
    espessura = max(2, largura // 400)
    for a, b in LIGACOES:
        cv2.line(quadro_bgr, pts[a], pts[b], cor, espessura, cv2.LINE_AA)
    for i, ponto in enumerate(pts):
        cv2.circle(quadro_bgr, ponto, espessura + 2, (255, 255, 255), -1, cv2.LINE_AA)
    if dedos is not None:
        for esticado, (_, _, _, ponta) in zip(dedos, DEDOS):
            if esticado:
                cv2.circle(quadro_bgr, pts[ponta], espessura * 4, (60, 200, 255), -1, cv2.LINE_AA)
    return quadro_bgr


class DecisorNumeros:
    """Decide os disparos dos números e diz quando a CNN pode disparar.

    Um número dispara quando a mesma pose fica estável por `tempo_estavel` segundos com a mão parada:
    o pulso não anda mais que `movimento` vezes o tamanho da palma, nem a mão muda de tamanho na imagem
    (aproximar ou afastar) mais que `escala` (em log). Para repetir o mesmo número, a pose precisa mudar
    ou a mão sair antes. A pose de cada quadro é a mais votada nos últimos `votos` quadros, para uma
    leitura errada isolada não interromper a contagem.
    """

    def __init__(self, tempo_estavel=0.5, movimento=0.5, escala=0.25, bloqueio=1.0, votos=5, rearmar=0.25):
        self.tempo_estavel, self.movimento, self.escala = tempo_estavel, movimento, escala
        self.bloqueio, self.rearmar = bloqueio, rearmar
        self.votos = collections.deque(maxlen=votos)
        self.historico = collections.deque()  # (tempo, pose, pulso_xy, tamanho)
        self.pose, self.dedos, self.mao = None, None, None
        self.inicio_pose, self.progresso, self.parado = 0.0, 0.0, False
        self.disparada, self.armado = None, True
        self.numeros_bloqueados_ate, self.cnn_bloqueada_ate = 0.0, 0.0

    def atualizar(self, mao, agora, largura=1.0, altura=1.0):
        """Recebe a mão do quadro (ou None) e devolve o número disparado, ou None."""
        self.mao = mao
        if mao is None:
            self.dedos, pose_quadro = None, None
        else:
            self.dedos = contar_dedos(mao.mundo)
            pose_quadro = classificar_pose(self.dedos)
        self.votos.append(pose_quadro)
        contagem = collections.Counter(self.votos)
        pose = pose_quadro if contagem[pose_quadro] * 2 > len(self.votos) else contagem.most_common(1)[0][0]

        if pose != self.pose:
            self.pose, self.inicio_pose = pose, agora
        duracao = agora - self.inicio_pose
        if not self.armado and (pose != self.disparada and duracao >= self.rearmar):
            self.armado = True

        if mao is not None:
            pulso = np.array([mao.imagem[0, 0] * largura, mao.imagem[0, 1] * altura])
            palma = np.array([mao.imagem[9, 0] * largura, mao.imagem[9, 1] * altura])
            tamanho = max(float(np.linalg.norm(palma - pulso)), 1e-6)
            self.historico.append((agora, pose, pulso, tamanho))
        while self.historico and agora - self.historico[0][0] > self.tempo_estavel:
            self.historico.popleft()

        self.parado = False
        if mao is not None and pose in NUMEROS and len(self.historico) >= 2:
            _, _, pulso, tamanho = self.historico[-1]
            andou = max(float(np.linalg.norm(h[2] - pulso)) for h in self.historico) / tamanho
            mudou = max(abs(math.log(h[3] / tamanho)) for h in self.historico)
            self.parado = andou <= self.movimento and mudou <= self.escala
        self.progresso = min(duracao / self.tempo_estavel, 1.0) if (pose in NUMEROS and self.parado) else 0.0

        if (pose in NUMEROS and self.parado and duracao >= self.tempo_estavel and self.armado
                and agora >= self.numeros_bloqueados_ate):
            self.armado, self.disparada = False, pose
            self.cnn_bloqueada_ate = agora + self.bloqueio
            return pose
        return None

    def cnn_pode_disparar(self, agora):
        """False enquanto a mão está parada numa pose de número, e logo depois de um número disparar."""
        segurando_numero = self.pose in NUMEROS and self.parado and agora - self.inicio_pose >= 0.3
        return not segurando_numero and agora >= self.cnn_bloqueada_ate

    def cnn_disparou(self, agora):
        self.numeros_bloqueados_ate = agora + self.bloqueio

    def confiancas(self):
        """Uma "confiança" por número, para as telas: o progresso da pose atual até disparar."""
        return {n: (self.progresso if n == self.pose else 0.0) for n in NUMEROS}
