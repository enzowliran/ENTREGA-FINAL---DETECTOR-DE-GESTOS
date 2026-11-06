import re, esprima
p = 'demo/player.py'
s = open(p, encoding='utf-8').read()
a = '''            elif url.path == "/eventos":
'''
b = '''            elif url.path.startswith("/vendor/"):
                # Só arquivos diretos de demo/vendor, sem subpastas.
                arquivo = RAIZ / "vendor" / Path(url.path).name
                if not arquivo.is_file():
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/javascript; charset=utf-8")
                self.end_headers()
                self.wfile.write(arquivo.read_bytes())
            elif url.path == "/eventos":
'''
assert a in s
s = s.replace(a, b, 1)
open(p, 'w', encoding='utf-8').write(s)
h = open('demo/tenis.html', encoding='utf-8').read()
js = re.findall(r'<script>(.*?)</script>', h, re.S)[0]
esprima.parseScript(js)
print('player ok; JS do jogo ok')
