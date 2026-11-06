from pathlib import Path

# 1. Separa as músicas sintetizadas em um arquivo usado pelo player e pelo painel.
p = Path('demo/player.html')
s = p.read_bytes().decode('utf-8')
ini_marca = '// Músicas originais, geradas no navegador'
fim_marca = 'function agendar() {'
ini, fim = s.index(ini_marca), s.index(fim_marca)
Path('demo/vendor/sintese_musicas.js').write_bytes(s[ini:fim].encode('utf-8'))
abre = s.rindex('<script>', 0, ini)
s = s[:abre] + '<script src="/vendor/sintese_musicas.js"></script>\n<script>\n' + s[fim:]
p.write_bytes(s.encode('utf-8'))

# 2. Servidor: pasta de músicas, com pedido parcial, e volume pelo polegar.
p = Path('demo/player.py')
s = p.read_bytes().decode('utf-8')
crlf = '\r\n' in s
s = s.replace('\r\n', '\n')
trocas = [
 ('COMANDOS = {"Swiping Left": "anterior", "Swiping Right": "proxima", "Stop Sign": "alternar"}\n',
  'COMANDOS = {"Swiping Left": "anterior", "Swiping Right": "proxima", "Stop Sign": "alternar"}\n'
  'COMANDOS_VOLUME = {"Thumb Up": "volume_mais", "Thumb Down": "volume_menos"}\n'
  'TIPOS_AUDIO = {".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".wav": "audio/wav", ".ogg": "audio/ogg"}\n'),
 ('''            elif url.path == "/eventos":
''', '''            elif url.path == "/musicas/lista":
                pasta = Path(args.musicas)
                nomes = sorted(a.name for a in pasta.iterdir() if a.suffix.lower() in TIPOS_AUDIO) if pasta.is_dir() else []
                corpo = json.dumps(nomes, ensure_ascii=False).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(corpo)
            elif url.path.startswith("/musicas/"):
                self.enviar_audio(Path(args.musicas) / Path(unquote(url.path)).name)
            elif url.path == "/eventos":
'''),
 ('''        def do_GET(self):
''', '''        def enviar_audio(self, arquivo):
            """Entrega o áudio, inclusive em partes, para o navegador poder avançar na faixa."""
            if arquivo.suffix.lower() not in TIPOS_AUDIO or not arquivo.is_file():
                self.send_response(404)
                self.end_headers()
                return
            tamanho = arquivo.stat().st_size
            inicio, fim, parcial = 0, tamanho - 1, False
            faixa = re.fullmatch(r"bytes=(\\d*)-(\\d*)", self.headers.get("Range", ""))
            if faixa and (faixa[1] or faixa[2]):
                parcial = True
                if faixa[1]:
                    inicio = int(faixa[1])
                    fim = min(int(faixa[2]), tamanho - 1) if faixa[2] else tamanho - 1
                else:
                    inicio = max(tamanho - int(faixa[2]), 0)
                if inicio > fim:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{tamanho}")
                    self.end_headers()
                    return
            self.send_response(206 if parcial else 200)
            self.send_header("Content-Type", TIPOS_AUDIO[arquivo.suffix.lower()])
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(fim - inicio + 1))
            if parcial:
                self.send_header("Content-Range", f"bytes {inicio}-{fim}/{tamanho}")
            self.end_headers()
            try:
                with arquivo.open("rb") as f:
                    f.seek(inicio)
                    restante = fim - inicio + 1
                    while restante > 0 and (bloco := f.read(min(65536, restante))):
                        self.wfile.write(bloco)
                        restante -= len(bloco)
            except OSError:
                pass

        def do_GET(self):
'''),
 ('import queue\n', 'import queue\nimport re\n'),
 ('from urllib.parse import parse_qs, urlparse\n', 'from urllib.parse import parse_qs, unquote, urlparse\n'),
 ('''    a.add_argument("--nao-abrir", action="store_true", help="Não abre o navegador")
''', '''    a.add_argument("--nao-abrir", action="store_true", help="Não abre o navegador")
    a.add_argument("--musicas", default=str(RAIZ / "musicas"), help="Pasta com os áudios servidos em /musicas/")
    a.add_argument("--volume", action="store_true", help="Polegar para cima e para baixo controlam o volume")
'''),
 ('''    args = a.parse_args()

    estado, parar''', '''    args = a.parse_args()
    if args.volume:
        COMANDOS.update(COMANDOS_VOLUME)

    estado, parar'''),
]
for a, b in trocas:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
p.write_bytes((s.replace('\n', '\r\n') if crlf else s).encode('utf-8'))

# 3. Atalho, pasta e git: os áudios enviados não entram no repositório.
Path('demo/abrir_painel.bat').write_bytes((
    '@echo off\r\nchcp 65001 >nul\r\ncd /d "%~dp0.."\r\n'
    'if not exist demo\\musicas mkdir demo\\musicas\r\n'
    '".venv\\Scripts\\python.exe" -W ignore demo\\player.py --pagina painel_carro.html --porta 8770 --volume %*\r\npause\r\n').encode())
Path('demo/musicas').mkdir(exist_ok=True)
g = Path('.gitignore')
t = g.read_bytes().decode('utf-8')
if '/demo/musicas/' not in t:
    g.write_bytes((t.rstrip() + '\n/demo/musicas/\n').encode('utf-8'))
print('ok')
