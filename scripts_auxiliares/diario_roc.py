import json
import uuid

p = 'notebook.ipynb'
nb = json.load(open(p, encoding='utf-8'))
texto = """### 05/11/2026: ROC e AUC do modelo atual

Medição por vídeo isolado (`src/avaliar_roc.py`), sem alterar o modelo nem escolher nada a partir do teste. A acurácia reproduzida conferiu com os valores anteriores.

| Medida | Validação | Teste |
|---|---:|---:|
| Acurácia | 93,3% | 92,6% |
| AUC gesto x sem gesto | 0,9977 | 0,9982 |
| AUC macro, 27 classes | 0,9978 | 0,9980 |
| Gestos detectados com 1% de falso positivo | 98,0% | 97,3% |
| Gestos detectados com 0,1% de falso positivo | 67,3% | 74,5% |

**Leitura:** a AUC alta mostra boa separação em vídeos isolados, mas não mede o uso contínuo. Com o gatilho da demo, a detecção é de 72,4% em clipes colados, e a contagem de falsos disparos por minuto continua sendo a métrica principal do trabalho. A AUC gesto x sem gesto também não verifica se o gesto foi classificado corretamente."""
indice = max(i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'markdown' and 'reforço de esquerda e direita, descartado' in ''.join(c['source']))
celula = {'cell_type': 'markdown', 'metadata': {}, 'source': texto.splitlines(True)}
if nb.get('nbformat_minor', 0) >= 5:
    celula['id'] = uuid.uuid4().hex[:8]
nb['cells'].insert(indice + 1, celula)
with open(p, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write('\n')
print('ok')
