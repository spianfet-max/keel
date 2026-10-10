import json,sys,os
S=json.load(open('samples.json'))
css=open('core.css').read(); js=open('core.js').read()
os.makedirs('dist',exist_ok=True)
for name,key in [('carry','fxcarry'),('brief','brief'),('barrier','sp'),('arcade','comps')]:
    p=f'{name}.src.html'
    if not os.path.exists(p): continue
    s=open(p).read().replace('/*CORE_CSS*/',css).replace('/*CORE_JS*/',js).replace('/*SAMPLE*/',json.dumps(S[key],separators=(',',':')))
    open(f'dist/{name}.html','w').write(s); print(name,len(s))
