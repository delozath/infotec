import pdb

import numpy   as np
import pandas  as pd
import seaborn as sns; sns.set(style="ticks", palette="pastel")

from matplotlib           import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

def two_bivariate_normals( **params ):
    X1 = np.random.multivariate_normal( params['M1'],params['S1'],params['N1'] )
    X2 = np.random.multivariate_normal( params['M2'],params['S2'],params['N2'] )
    
    Y      = -np.ones( params['N1'] + params['N2'])
    
    Y[ params['N1']: ] = 1
    Y                  = Y[:,np.newaxis]
    
    data = np.concatenate( (X1  ,X2),axis=0 )
    data = np.concatenate( (data, Y),axis=1 )
    
    data = pd.DataFrame(data,columns=['X_1','X_2','Y'])
    return data.copy()

# TODO: Agregar los otros generadores de datos del jupyter notebook

def main(fname, **params ):
    data = two_bivariate_normals( **params )
    sns.scatterplot( x='X_1', y='X_2', hue='Y', data=data,
                      palette=['b','k'] )
    
    pdb.set_trace()
    data.to_csv(fname)   
    pdb.set_trace()

"""
ax.scatter(xs, ys, zs, marker=m)


"""

if __name__ == '__main__':
    M_C1 = np.array([    1.0,  0.0 ])
    S_C1 = np.array([ [  1.0,  0.0 ],
                      [  0.0,  1.0 ] ])
    
    M_C2 = np.array([   -3.0,  2.0 ])
    S_C2 = np.array([ [  2.0, -0.5 ],
                      [ -0.5,  2.0 ] ])
    
    N_C1 = 300
    N_C2 = 300
    
    PATH  = '/home/omarpr/git/machine_learning/data/'
    fname = 'two_bivariate_normals.csv'
    main( PATH+fname,
          M1=M_C1, S1=S_C1, N1=N_C1,
          M2=M_C2, S2=S_C2, N2=N_C2 )