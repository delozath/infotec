import pdb

import numpy   as np
import pandas  as pd
import seaborn as sns; sns.set(style="ticks", palette="pastel")

from matplotlib           import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

def pseudo_inv (X: np.array, Y: np.array)->np.array:
    """
      Compute the pseudoinverse matrix for regression/classification
        -> pinv = (X^T X)^-1 X^T y
        
      @param X: -> shape = (n_samples,k_features)
      @param Y: -> shape = (n_samples,1)
      
      @return pinv: -> shape = (k_features,1)
    """
    pinv = np.dot( X.T,X )
    pinv = np.linalg.inv(pinv)
    pinv = np.dot( pinv,X.T )
    pinv = np.dot( pinv,Y   )
    
    return pinv

def extend_x(X: np.array)->np.array:
    """
      Appends a colum of ones to the X matrix
        -> X_ex = [X Ones]
      
      @param X: -> shape = (n_samples,k_features)
      
      @return X_ex: -> shape = (n_samples,k_features+1)
    """
    ones = np.ones( X.shape[0] )[:,np.newaxis]
    
    return np.concatenate( (X,ones), axis=-1 )

def main( **params ):
    
    """
    data = pd.read_csv( params['fname'] )
    
    sns.scatterplot( x='X_1', y='X_2', hue='Y', data=data,
                      palette=['b','k'] )
    
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    
    mask = data['Y'] == -1
    ax.scatter( data['X_1'][ mask], data['X_2'][ mask], data['Y'][ mask], color='b' )
    ax.scatter( data['X_1'][~mask], data['X_2'][~mask], data['Y'][~mask], color='k' )
    
    ax.view_init(azim=-90, elev=90)
    ax.set_xlabel('X_1')
    ax.set_ylabel('X_2')
    ax.set_zlabel('Y')
    
    point  = np.array([1, 0, -1])
    normal = np.array([1, -1, 2])
    
    d = -point.dot(normal)
    
    xx, yy = np.meshgrid( np.linspace( data['X_1'].min(),data['X_1'].max(),15 ),
                          np.linspace( data['X_2'].min(),data['X_2'].max(),15 )  )
    
    z = (-normal[0] * xx - normal[1] * yy - d) / normal[2]
    
    
    ax.plot_surface( xx, yy, z, cmap='inferno', facecolors=plt.cm.inferno(  (z - z.min() )/( z-z.min() ).max()  ), alpha=.4 )

    plt.show()
    
    pdb.set_trace()

   """


"""

if __name__ == '__main__':
    PATH  = '/home/omarpr/git/machine_learning/data/'
    fname = 'two_bivariate_normals.csv'
    
    main( fname=PATH+fname  )