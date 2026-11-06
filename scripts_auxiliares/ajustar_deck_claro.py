trocas = {
 'demo/gerar_apresentacao.py': [
  ('if not melhores or l["det"] > melhores[-1]["det"]:', 'if not melhores or l["det"] * 100 > melhores[-1]["det"]:'),
 ],
 'demo/apresentacao_modelo.html': [
  ('95% de acerto, dispara sozinho', 'Acerta muito, mas dispara sozinho'),
  ('No volante, sem ajuste, o modelo não reconhece nada', 'No volante, sem ajuste: zero acertos'),
  ('L = { e: 58, d: 120, t: 16, b: 60, w: 760, h: 410 }', 'L = { e: 58, d: 170, t: 16, b: 60, w: 760, h: 410 }'),
  ('L = { e: 72, d: 24, t: 18, b: 70, w: 800, h: 430 }', 'L = { e: 104, d: 24, t: 18, b: 76, w: 800, h: 430 }'),
  ('y: L.h - 2, fill: "#5a5f64", "font-size": 20, "text-anchor": "middle" }, "Falsos disparos', 'y: L.h - 10, fill: "#5a5f64", "font-size": 20, "text-anchor": "middle" }, "Falsos disparos'),
  ('[[0.01, "0,01"], [0.1, "0,1"], [1, "1"]]', '[[minimo, "0"], [0.01, "0,01"], [0.1, "0,1"], [1, "1"]]'),
 ],
}
for p, lista in trocas.items():
    s = open(p, encoding='utf-8').read()
    for a, b in lista:
        assert a in s, a[:60]
        s = s.replace(a, b)
    open(p, 'w', encoding='utf-8').write(s)
print('ok')
