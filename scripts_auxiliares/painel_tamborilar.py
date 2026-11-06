import re
from pathlib import Path

p = Path('demo/player.py')
s = p.read_bytes().decode('utf-8')
a = '"Shaking Hand": "acenar"'
assert s.count(a) == 1
s = s.replace(a, '"Drumming Fingers": "tamborilar"')
s = s.replace('polegares e aceno', 'polegares e tamborilar')
p.write_bytes(s.encode('utf-8'))

p = Path('demo/painel_carro.html')
s = p.read_bytes().decode('utf-8')
n = len(re.findall('acenar|Acenar', s))
s = s.replace('acenar', 'tamborilar').replace('Acenar', 'Tamborilar os dedos')
assert not re.search('acen|Acen|aceno', s)
p.write_bytes(s.encode('utf-8'))
print('trocas no painel:', n)
