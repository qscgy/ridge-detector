import os, pickle

with open('test_set.pkl','rb') as f:
    data = pickle.load(f)

fixed = []
for k in data:
    fk = k[len('/playpen/Datasets/geodepth2/'):]
    fk = k.replace('img_corr','image')
    fixed.append(fk)
print(fixed)

with open('test_set_rel.pkl','wb') as f:
    pickle.dump(fixed, f)