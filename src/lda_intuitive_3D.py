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
    
    return np.concatenate( (X,ones), axis=-1 ).copy()

def plot_2D_classes( X, Y ):
    plt.figure()
    sns.scatterplot( x=X[:,0], y=X[:,1], hue=Y, palette=['forestgreen','darkblue'])
    
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    
    mask = Y == -1
    ax.scatter( X[:,0][ mask], X[:,1][ mask], Y[ mask], color='forestgreen' )
    ax.scatter( X[:,0][~mask], X[:,1][~mask], Y[~mask], color='darkblue' )
    
    ax.view_init(azim=-90, elev=90)
    ax.set_xlabel('X_1')
    ax.set_ylabel('X_2')
    ax.set_zlabel('Y')
    
    plt.show()

def plot_decision_plane( X, Y, W ):
    plt.figure()
    sns.scatterplot( x=X[:,0], y=X[:,1], hue=Y, palette=['forestgreen','darkblue'])
    
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    
    mask = Y == -1
    ax.scatter( X[:,0][ mask], X[:,1][ mask], Y[ mask], color='forestgreen' )
    ax.scatter( X[:,0][~mask], X[:,1][~mask], Y[~mask], color='darkblue' )
    
    ax.view_init(azim=-90, elev=90)
    ax.set_xlabel('X_1')
    ax.set_ylabel('X_2')
    ax.set_zlabel('Y')
    
    xx, yy = np.meshgrid( np.linspace( X[:,0].min(),X[:,0].max(),15 ),
                          np.linspace( X[:,1].min(),X[:,1].max(),15 )  )
    sh = np.array( xx.shape ) 
    mesh = np.concatenate( (xx.reshape( np.prod(sh),1 ), yy.reshape( np.prod(sh),1 ) ),axis=1 )
    # TODO: extender signo
    pdb.set_trace()
    plt.show()
    
def threshold(data,thre=0.0,comparison=0):
    if comparison==0:
        l = data >= thre
    else:
        l = data < thre
    return (-2*l+1).copy()

def pseudoinverse_train(X,Y):
    X_ext = extend_x(X)
    W     = pseudo_inv(X_ext, Y)
    
    return W.copy()

def main( **params ):
    data = pd.read_csv( params['fname_uninormals'] )
    X    = data.loc[ :,['X_1','X_2'] ].values
    Y    = data['Y'  ].values.astype('int')
    
    plot_2D_classes( X, Y )
    
    N     = 200
    P     = X.shape[0]-N
    index = np.arange( X.shape[0] ); np.random.shuffle(index)
    
    X_Train, Y_Train = X[ index[  :N] ], Y[ index[  :N] ]
    X_Test , Y_Test  = X[ index[-P: ] ], Y[ index[-P: ] ]
    
    W = pseudoinverse_train(X,Y)
    
    plot_decision_plane( X,Y,W )
    pdb.set_trace()
    empiric_decision( X,Y, 
                        np.concatenate( (lines,W.T) ), 
                        thre + [0]                   ,
                        sign + [0]
                    )
    
    planes_predict( data, 'Un Rasgo, Dos clases', W.T )
    
    ecs = ["$$y(x) = %.1fx %+ .1f$$"%(tuple(i)) for i in lines]
    for e in ecs:
        print(e)
    
    pdb.set_trace()

def _3d(**params):
    data = pd.read_csv( params['fname_uninormals'] )
    
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




if __name__ == '__main__':
    PATH             = '/home/omarpr/git/machine_learning/data/'
    fname_binormals  = 'CNIB 2020 TWO BIV NORMALS.csv'
    
    main( fname_uninormals=PATH+fname_binormals  )
    #_3d(fname_uninormals=PATH+fname_binormals)