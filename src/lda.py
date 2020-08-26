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

def plot_varrows( x,y1,y2,alpha=1,color='black' ):
    for i,j,k in zip( x,y1,y2 ):
        plt.arrow( i,j, 0,k-j,alpha=alpha,color=color )

def motivation( data,experiment ):
    title = experiment.replace(', ','-')
    y     = np.zeros( data.shape[0] )
    
    plt.figure( title )
    sns.scatterplot( x='X', y=y, hue='Y', data=data,
                      palette=['tomato','forestgreen'],
                      alpha=.8)
    plt.title(title)
    
    plt.figure( title + ' etiquetas' )
    sns.scatterplot( x='X', y='Y', hue='Y', data=data,
                      palette=['tomato','forestgreen'],
                      alpha=.8)
    plt.title( title +' etiquetas')
    
    plt.show()
    

def planes(data,experiment,lines):
    title    = experiment.replace(', ','-') + u' planos separación'
    nlines   = lines.shape[0]
    min, max = data['X'].min(), data['X'].max()
    min, max = min + 2*min , max + 0.1*max
    
    xline  = np.linspace( min,max,100 )[np.newaxis]
    ylines = np.dot( lines[:,0:1],xline ).T
    ylines = ylines + lines[:,1]
    
    plt.figure( title )
    sns.scatterplot( x='X', y='Y', hue='Y', data=data,
                      palette=['tomato','forestgreen'],
                      alpha=.8)
    plt.plot( np.tile( xline,(nlines,1) ).T, ylines)
    plt.title(title)
    
    plt.show()

def planes_predict(data,experiment,lines):
    title    = experiment.replace(', ','-') + u' planos predicción'
    x = data['X'].values[np.newaxis]
    y = np.dot( lines[:,0:1],x ).T
    y = y + lines[:,1]
    
    plt.figure( title )
    plot_varrows( x.ravel(),
                  data['Y'].values,    
                  y[:,0],
                  alpha=1,color='black' )
    
    sns.scatterplot( x='X', y='Y', hue='Y', data=data,
                      palette=['tomato','forestgreen'],
                      alpha=.8)
    plt.plot( x.T, y)
    plt.title(title)
    
    plt.show()

def threshold(data,thre=0.0,comparison=0):
    if comparison==0:
        l = data > thre
    else:
        l = data < thre
    
    
    return (-2*l+1).copy()
    

def main( **params ):
    lines = np.array([ [-0.2  , 0.9  ],
                       [-0.1  , 0.3  ],
                       [ 0.0  , 0.3  ],
                       [ 0.1  ,-0.3  ],
                       [ 0.4  ,-0.8  ] ])                   
    data  = pd.read_csv( params['fname_uninormals'] )
    
    #motivation    (data,'Un Rasgo, Dos clases')
    #planes        (data,'Un Rasgo, Dos clases',lines)
    #planes        (data,'Un Rasgo, Dos clases',lines[ [0,-1] ])
    #planes_predict( data, 'Un Rasgo, Dos clases', lines[ [0] ])
    #planes_predict( data, 'Un Rasgo, Dos clases', lines[ [1] ])
    #planes_predict( data, 'Un Rasgo, Dos clases', lines[ [2] ])
    #planes_predict( data, 'Un Rasgo, Dos clases', lines[ [3] ])
    #planes_predict( data, 'Un Rasgo, Dos clases', lines[ [4] ])
    
    x = data['X'].values[np.newaxis]
    y = np.dot( lines[:,0:1],x ).T
    y = y + lines[:,1]
    
    Y = list(  map( lambda z: threshold(z[0],z[1],z[2]), zip( y.T,[.5, .1, .3, -0.15, 0], [0,0,0,-1,-1] )  )  )
    Y = np.array(Y).T
    
    plt.pcolormesh(Y==-data['Y'].values[:,np.newaxis],edgecolors='k', linewidth=.5,cmap='cool')
    plt.xticks(np.arange(Y.shape[1])+.5,['Modelo 1','Modelo 2','Modelo 3','Modelo 4','Modelo 5'], fontsize=10)
    
    ax = plt.gca()
    #ax.xticklabels()
    ax.set_aspect(.05)
    plt.gca().invert_yaxis()
    plt.show()
    print(np.array(Y).T)
    pdb.set_trace()
    ecs = ["$$y(x) = %.1fx %+ .1f$$"%(tuple(i)) for i in lines]
    for e in ecs:
        print(e)
    
    
    
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