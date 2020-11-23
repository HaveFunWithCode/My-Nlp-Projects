with open ('Files/best.txt') as best:
    best=best.read().split('\n')
with open ('Files/keywords-raw.txt') as f:
    keys=f.read().split('\n')

fills=[a for a in best if a in keys]

with open ('Files/keywords.txt','w') as f2:
    best=f2.write('\n'.join(fills))
