import sys,pypdf
for f in sys.argv[1:]:
    r=pypdf.PdfReader(f); print('=====',f[-22:],len(r.pages),'paginas')
    print('\n'.join(p.extract_text() or '' for p in r.pages)[:9000])
