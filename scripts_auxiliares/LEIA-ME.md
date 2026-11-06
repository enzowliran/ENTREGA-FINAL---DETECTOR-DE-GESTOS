# Scripts auxiliares

Scripts avulsos usados ao longo do projeto, guardados como registro. O código principal está em `src/` e `demo/`. Rode a partir da raiz do repositório.

**Análises que ainda podem ser usadas:**

| Script | O que faz |
|---|---|
| `comparar_global.py` | Compara limiares globais e janelas seguidas, usando `data/interno/janelas_validacao_cnn_3d.npz` |
| `confusao_lados.py` | Mede a confusão entre esquerda e direita; precisa do Jester |
| `ver_gestos.py` | Monta tiras de frames do DriverMHG; precisa do dataset, e as imagens não devem ser publicadas aqui |
| `teste_gpu.py` | Testa a GPU |
| `teste_player.py` | Testa a demo do player |

**Os demais são registro.** Os que começam com `ajustar_`, `editar_`, `aplicar_`, `painel_`, `preparar_` e `diario_` e os scripts `montar_notebook.py` e `criar_nb01.py` foram edições feitas uma vez nos arquivos do projeto e já estão aplicadas. Não rode de novo: a maioria falha ao não achar o texto original, e alguns podem duplicar conteúdo no notebook.
