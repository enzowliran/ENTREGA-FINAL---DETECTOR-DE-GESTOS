p = 'demo/player.py'
s = open(p, encoding='utf-8').read()
a = '''    html = (RAIZ / "player.html").read_bytes()
'''
b = '''    html = (RAIZ / args.pagina).read_bytes()
'''
assert a in s; s = s.replace(a, b)
a = '''    a.add_argument("--porta", type=int, default=8765)
'''
b = '''    a.add_argument("--porta", type=int, default=8765)
    a.add_argument("--pagina", default="player.html", help="Página de demo/ a servir")
'''
assert a in s; s = s.replace(a, b)
open(p, 'w', encoding='utf-8').write(s)
import re, esprima
h = open('demo/apresentacao.html', encoding='utf-8').read() if False else None
print('ok')
