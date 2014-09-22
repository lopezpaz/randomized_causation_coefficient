PATH_X_TR = "syn_pairs.csv"
PATH_Y_TR = "syn_target.csv"
PATH_X_TE = "tuebingen_pairs.csv"
PATH_Y_TE = "tuebingen_target.csv"

import numpy as np
from   sklearn.ensemble         import RandomForestClassifier as RFC
from   sklearn.preprocessing    import scale

def featurize_row(row,w,i,j):
  r  = row.split(",",2)
  x  = scale(np.array(r[i].split(),dtype=np.float))
  y  = scale(np.array(r[j].split(),dtype=np.float))
  b  = np.ones(x.shape)  
  dx = np.cos(np.dot(w2,np.vstack((x,b)))).mean(1)
  dy = np.cos(np.dot(w2,np.vstack((y,b)))).mean(1)
  if(sum(dx) > sum(dy)):
    return np.hstack((dx,dy,np.cos(np.dot(w,np.vstack((x,y,b)))).mean(1)))
  else:
    return np.hstack((dx,dy,np.cos(np.dot(w,np.vstack((y,x,b)))).mean(1)))

def featurize(x,w,flip):
  f1  = np.array([featurize_row(row,w,1,2) for row in x])
  if(flip==1):
    return np.vstack((f1,np.array([featurize_row(row,w,2,1) for row in x])))
  else:
    return f1

def read_pairs(filename):
  f = open(filename);
  pairs = f.readlines();
  f.close();
  del pairs[0];
  return pairs

k    = 100
s    = 10

np.random.seed(0);
w  = np.hstack((s*np.random.randn(k,2),2*np.pi*np.random.rand(k,1)))
w2 = np.hstack((s*np.random.randn(k,1),2*np.pi*np.random.rand(k,1)))
y = np.genfromtxt(PATH_Y_TR, delimiter=",")[:,1]
x = read_pairs(PATH_X_TR)
x = featurize(x,w,1)
y = np.hstack((y,-y))

reg = RFC(verbose=10,random_state=0, max_features=None,
max_depth=None,min_samples_leaf=100, n_estimators=500, n_jobs=4).fit(x,y);

x_te = read_pairs(PATH_X_TE)
x_te = featurize(x_te,w,0)
y_te = 2*((np.genfromtxt(PATH_Y_TE,delimiter=",")[:,1])==1)-1
m_te = np.genfromtxt(PATH_Y_TE,delimiter=",")[:,2]

print(sum((reg.predict(x_te)==y_te)*m_te)/sum(m_te))
