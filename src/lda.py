import pdb

import numpy   as np
import pandas  as pd
import seaborn as sns; sns.set(style="ticks", palette="pastel")

from matplotlib           import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use({  'figure.figsize'    :(12,6),
                 'axes.titlesize'    :20,
                 'axes.titleweight'  :True,
                 'lines.markersize'  :10,
                 'axes.grid'         :True,
                 'axes.labelsize'    :16,
                 'xtick.labelsize'   :14,
                 'xtick.major.width' :True,
                 'ytick.labelsize'   :14,
                 'ytick.major.width' :True, 
                 'lines.linewidth'   :2.5   })

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

def motivation( data,experiment ):
    
    
    y = np.zeros( data.shape[0] )
    plt.figure( experiment.replace(', ','-') )
    sns.scatterplot( x='X', y=y, hue='Y', data=data,
                      palette=['tomato','forestgreen'],
                      alpha=.8)
    plt.title(experiment)
    
    plt.figure( experiment.replace(', ','-')+' etiquetas' )
    sns.scatterplot( x='X', y='Y', hue='Y', data=data,
                      palette=['tomato','forestgreen'],
                      alpha=.8)
    plt.title(experiment +' etiquetas')
    
    plt.show()
    

def planes(data,experiment,nlines,bias):
    xline  = np.linspace( 1.2*data['X'].min(), 1.2*data['X'].max(), 100 )[np.newaxis]
    
    pdb.set_trace()
    W      = np.linspace(-.75,.75,nlines)[:,np.newaxis]
    B      = -bias*W
    ylines = np.dot(W,xline)
        
    plt.figure( experiment.replace(', ','-')+' etiquetas' )
    sns.scatterplot( x='X', y='Y', hue='Y', data=data,
                      palette=['tomato','forestgreen'],
                      alpha=.8)
    plt.plot( np.tile( xline,(nlines,1) ).T, ylines.T+B.T)
    plt.title(experiment + ' etiquetas')
    
    plt.show()

def main( **params ):
    data = pd.read_csv( params['fname_uninormals'] )
    #motivation(data,'Un Rasgo, Dos clases')
    planes    (data,'Un Rasgo, Dos clases',5,2.5)
    
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

if __name__ == '__main__':
    PATH             = '/home/omarpr/git/machine_learning/data/'
    fname_uninormals = 'CNIB 2020 TWO UNIV NORMALS.csv'
    
    main( fname_uninormals=PATH+fname_uninormals  )