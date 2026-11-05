"""Redes de vídeo treinadas do zero, com inicialização aleatória e sem pesos de terceiros.

Entrada: tensor (lote, tempo, canais, altura, largura).
"""
import torch
from torch import nn


def _bloco_2d(entrada, saida):
    return nn.Sequential(nn.Conv2d(entrada, saida, 3, padding=1, bias=False), nn.BatchNorm2d(saida), nn.ReLU(inplace=True),
                         nn.Conv2d(saida, saida, 3, padding=1, bias=False), nn.BatchNorm2d(saida), nn.ReLU(inplace=True),
                         nn.MaxPool2d(2))


class CNNGRU(nn.Module):
    """CNN 2D aplicada a cada frame, seguida de uma GRU sobre a sequência."""

    def __init__(self, n_classes, canais_entrada=3, canais=(32, 64, 128, 256), oculta=256, dropout=0.3):
        super().__init__()
        camadas, entrada = [], canais_entrada
        for saida in canais:
            camadas.append(_bloco_2d(entrada, saida))
            entrada = saida
        self.corpo = nn.Sequential(*camadas, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.gru = nn.GRU(entrada, oculta, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.cabeca = nn.Linear(oculta, n_classes)

    def forward(self, x):
        lote, tempo = x.shape[:2]
        h = self.corpo(x.flatten(0, 1)).view(lote, tempo, -1)
        saida, _ = self.gru(h)
        return self.cabeca(self.dropout(saida.mean(dim=1)))


class CNNTransformer(nn.Module):
    """A mesma CNN 2D por frame da CNNGRU, com atenção temporal no lugar da GRU."""

    def __init__(self, n_classes, canais_entrada=3, canais=(32, 64, 128, 256), camadas_atencao=2,
                 cabecas=4, n_frames=16, dropout=0.3):
        super().__init__()
        blocos, entrada = [], canais_entrada
        for saida in canais:
            blocos.append(_bloco_2d(entrada, saida))
            entrada = saida
        self.corpo = nn.Sequential(*blocos, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.posicao = nn.Parameter(torch.zeros(1, n_frames, entrada))
        nn.init.trunc_normal_(self.posicao, std=0.02)
        camada = nn.TransformerEncoderLayer(entrada, cabecas, dim_feedforward=2 * entrada, dropout=0.1,
                                            batch_first=True, norm_first=True)
        self.atencao = nn.TransformerEncoder(camada, camadas_atencao, enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)
        self.cabeca = nn.Linear(entrada, n_classes)

    def forward(self, x):
        lote, tempo = x.shape[:2]
        h = self.corpo(x.flatten(0, 1)).view(lote, tempo, -1)
        h = self.atencao(h + self.posicao[:, :tempo])
        return self.cabeca(self.dropout(h.mean(dim=1)))


class CNN3D(nn.Module):
    """Convolução no espaço e no tempo. O tempo só é reduzido nas camadas finais."""

    def __init__(self, n_classes, canais_entrada=3, canais=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        camadas, entrada = [], canais_entrada
        for i, saida in enumerate(canais):
            passo_tempo = 2 if i >= 1 else 1
            camadas += [nn.Conv3d(entrada, saida, 3, padding=1, bias=False), nn.BatchNorm3d(saida), nn.ReLU(inplace=True),
                        nn.Conv3d(saida, saida, 3, padding=1, bias=False), nn.BatchNorm3d(saida), nn.ReLU(inplace=True),
                        nn.MaxPool3d((passo_tempo, 2, 2))]
            entrada = saida
        self.corpo = nn.Sequential(*camadas, nn.AdaptiveAvgPool3d(1), nn.Flatten())
        self.dropout = nn.Dropout(dropout)
        self.cabeca = nn.Linear(entrada, n_classes)

    def forward(self, x):
        return self.cabeca(self.dropout(self.corpo(x.transpose(1, 2))))


def criar_modelo(nome, n_classes, canais_entrada=3):
    if nome == "cnn_gru":
        return CNNGRU(n_classes, canais_entrada)
    if nome == "cnn_transformer":
        return CNNTransformer(n_classes, canais_entrada)
    if nome == "cnn_3d":
        return CNN3D(n_classes, canais_entrada)
    raise ValueError(nome)


def contar_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters())


def adaptar_entrada(modelo, canais_novos):
    """Troca os canais da primeira convolução, repetindo a média dos pesos aprendidos."""
    primeira = next(m for m in modelo.modules() if isinstance(m, (nn.Conv2d, nn.Conv3d)))
    with torch.no_grad():
        media = primeira.weight.mean(dim=1, keepdim=True)
        forma = [1] * media.dim()
        forma[1] = canais_novos
        primeira.weight = nn.Parameter(media.repeat(*forma) * primeira.in_channels / canais_novos)
    primeira.in_channels = canais_novos
    return modelo
