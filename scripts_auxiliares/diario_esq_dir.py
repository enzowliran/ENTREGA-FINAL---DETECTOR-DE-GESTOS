import json
import uuid

p = 'notebook.ipynb'
nb = json.load(open(p, encoding='utf-8'))
texto = """### 05/11/2026: reforço de esquerda e direita, descartado

Ajuste fino de 4 épocas a partir de `cnn_3d/melhor.pt` (`src/ajustar_esquerda_direita.py`), com taxa de aprendizado de 1e-4, cada vídeo de deslizar para a esquerda e a direita aparecendo 2 vezes por época e 20 mil vídeos das outras classes sorteados. Regra definida antes do treino: só aceitar épocas que perdessem no máximo 0,2 ponto no acerto geral da validação.

| Validação | Modelo atual | Melhor resultado do ajuste (época 4) |
|---|---:|---:|
| Acerto geral | 93,3% | 92,1% |
| Deslizar para a esquerda | 94,7% | 97,2% |
| Deslizar para a direita | 95,1% | 97,7% |
| Esquerda vira direita | 1,4% | 1,6% |
| Direita vira esquerda | 1,0% | 1,9% |

**Conclusão:** nenhuma época passou na regra. A rede passou a prever mais esses dois gestos, às custas das outras classes, e a troca entre os lados aumentou em vez de cair. Como nenhuma época foi aceita, os pesos não foram salvos e o fluxo simulado não foi medido. **Decisão:** a demo continua com `cnn_3d/melhor.pt`."""
indice = max(i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'markdown' and 'limiar por gesto, sem ganho real' in ''.join(c['source']))
celula = {'cell_type': 'markdown', 'metadata': {}, 'source': texto.splitlines(True)}
if nb.get('nbformat_minor', 0) >= 5:
    celula['id'] = uuid.uuid4().hex[:8]
nb['cells'].insert(indice + 1, celula)
with open(p, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write('\n')
print('celula', indice + 1, 'de', len(nb['cells']))
