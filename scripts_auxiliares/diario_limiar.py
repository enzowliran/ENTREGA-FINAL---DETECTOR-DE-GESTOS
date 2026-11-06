import json
import uuid

p = 'notebook.ipynb'
nb = json.load(open(p, encoding='utf-8'))
texto = """### 05/11/2026: limiar por gesto, sem ganho real

Teste na validação inteira do Jester (`src/calibrar_por_gesto.py`): 100,2 min sem gesto e 12.786 vídeos com gesto, usando as mesmas 3 janelas seguidas. O limiar por gesto foi avaliado em validação cruzada de 5 dobras. O conjunto de teste não foi usado.

| Gatilho | Falsos em 100 min | Gestos detectados | Disparos com classe errada |
|---|---|---:|---:|
| Limiar único 0,7 | 3 | 72,4% | 5,1% |
| Limiar único 0,6 | 5 | 78,3% | 7,3% |
| Limiar por gesto (fora da amostra) | 6 | 79,0% | 7,3% |
| Limiar único 0,55 | 6 | 80,3% | 8,4% |

**Conclusão:** o ganho de 6,5 pontos na detecção veio junto com o dobro de falsos disparos e mais erros de classe. Baixar o limiar único chega ao mesmo ponto, então a complexidade não se paga. A contagem de falsos é pequena e tem muita incerteza. As linhas de limiar único não passaram por validação cruzada, mas também não foram escolhidas nos dados. **Decisão:** a demo continua com o limiar único de 0,7."""
indice = max(i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'markdown' and 'modelo do Jester no DriverMHG, sem ajuste' in ''.join(c['source']))
celula = {'cell_type': 'markdown', 'metadata': {}, 'source': texto.splitlines(True)}
if nb.get('nbformat_minor', 0) >= 5:
    celula['id'] = uuid.uuid4().hex[:8]
nb['cells'].insert(indice + 1, celula)
with open(p, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write('\n')
print('celula', indice + 1, 'de', len(nb['cells']))
