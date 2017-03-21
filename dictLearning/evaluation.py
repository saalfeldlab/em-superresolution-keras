import numpy as np

import h5py
import scipy
import spams
import time
import scipy.ndimage.filters as sp
from scipy.interpolate import interp1d 

import patches

def upsampEval( X, dsfactor, patchSize, patchFnGrp=None, kind='avg' ):
    if dsfactor is not None:
        X_useme,dsz  = downsamplePatchList( X, patchSize, dsfactor, kind=kind )

        if patchFnGrp:
            patchFnGrp.create_dataset('patchesDown2', data=X_useme)

    Xre = upsamplePatchList( X_useme, dsz, dsfactor )

    if patchFnGrp:
        patchFnGrp.create_dataset('patchesUpsampled', data=Xre)

    xd = X - Xre 
    R = np.mean( (xd * xd).sum(axis=0))

    return R

def dictEval( X, D, param, lam=None, dsfactor=None, patchSize=None, patchFnGrp=None, kind='avg'):
    if dsfactor is not None:
        X_useme,dsz  = downsamplePatchList( X, patchSize, dsfactor, kind=kind )
        D_useme,Ddsz = downsamplePatchList( D, patchSize, dsfactor, kind=kind )

        if patchFnGrp:
            patchFnGrp.create_dataset('patchesDown', data=X_useme)
    else:
        X_useme = X
        D_useme = D

    if lam is None:
        lam = param['lambda1']

    #alpha = spams.lasso( np.asfortranarray(X_useme), D = np.asfortranarray(D_useme), **param )
    alpha = spams.lasso( np.asfortranarray(X_useme), D = np.asfortranarray(D_useme), return_reg_path = False,  **param )
    Xre = ( D * alpha )

    if patchFnGrp:
        patchFnGrp.create_dataset('patchesRecon', data=Xre)

    xd = X - Xre 

    R = np.mean( (xd * xd).sum(axis=0))

    if lam > 0:
        print "   dictEval - lambda: ", lam
        R = R + lam * np.mean( np.abs(alpha).sum(axis=0))

    return R

def dictUpsampleImage( im, D, param, dsfactor, patchSize=None,
        batchSize=500, stepSize=None ):

    fill_sub_patch = False
    offset = []
    if not stepSize==None:
        print( 'doing step size' )
        fill_sub_patch = True
        offset = np.floor( (np.array(patchSize) - np.array(stepSize)) / 2.) 
        slc = []
        for i in range( patchSize.size ):
            slc += [ slice( offset[i], offset[i] + stepSize[i] ) ]
    else:
        stepSize = patchSize


    xr = np.arange( patchSize[0], im.shape[0]-patchSize[0], stepSize[0] )
    yr = np.arange( patchSize[1], im.shape[1]-patchSize[1], stepSize[1] )
    zr = np.arange( patchSize[2], im.shape[2]-patchSize[2], stepSize[2] )

    x,y,z = np.meshgrid( xr, yr, zr )
    coords = np.array( [x.flatten(), y.flatten(), z.flatten()])
    N = coords.shape[1]

    padparams = np.zeros( [3,2], dtype='int' )
    padparams[ 2, 1 ] = dsfactor - ( im.shape[ 2 ] % dsfactor )
    impad = np.pad( im, padparams, 'constant' )

    patchSizeDown = np.array( patchSize ) / np.array( [ 1, 1, dsfactor ])

    tic = time.time()
    imRecon = np.zeros( im.shape, dtype = im.dtype )

    # imds = downsamplePatch3dAvgZ( impad, dsfactor )
    D_useme,Ddsz = downsamplePatchList( D, patchSize, [1, 1, dsfactor], kind='avg' )

    for i in range( 0, N, batchSize ):

        print( 'batch ', i, ' of ', N  )

        maxidx = np.min( [ N, i + batchSize ] )

        positions = coords[ :, i : maxidx ]

        X = patches.getPatchesCoordsEven( im, patchSize, positions )
        Xds,face = downsamplePatchList( X, patchSize, [ 1, 1, dsfactor], kind='avg' )

        # compute dictionary coefficients using low-res image and 
        alpha = spams.lasso( np.asfortranarray( Xds ).astype(float), D=np.asfortranarray(D_useme).astype(float),  return_reg_path = False, **param )
    
        # compute hi-res image with hi-res dictionary
        Xre = ( D * alpha )

        for j in range( positions.shape[1] ):
            x = positions[ 0, j ]
            y = positions[ 1, j ]
            z = positions[ 2, j ]

            if fill_sub_patch:
                imRecon[    x + offset[0] : x + offset[0] + stepSize[ 0 ],  
                            y + offset[1] : y + offset[1] + stepSize[ 1 ],  
                            z + offset[2] : z + offset[2] + stepSize[ 2 ]] = Xre[:,j].reshape( patchSize )[ slc ]
            else:
                imRecon[    x : x + patchSize[ 0 ],  
                            y : y + patchSize[ 1 ],  
                            z : z + patchSize[ 2 ]] = Xre[:,j].reshape( patchSize )

    toc = time.time()
    t = toc - tic
    print 'time to compute upsampling %f' % t
    return imRecon

def downsamplePatchList( patchList, patchSize, factor, kind='avg' ):
    (M,N) = patchList.shape

    # check that patch vector length is consistent w/ patchSize
    if np.prod( patchSize ) != M:
        print "downsamp - inconsistent patch size"
        return None,None

    for i in range(0,N):

        patch = np.reshape( patchList[:,i], patchSize )
        patch_ds = downsamplePatch3d( patch, factor, 0.5, 0.5, kind=kind )

        if i == 0:
            patchOutSz = patch_ds.shape
            patchOut = np.zeros( [np.prod(patchOutSz), N] )

        patchOut[:,i] = patch_ds.flatten() 

    return patchOut, patchOutSz

def upsamplePatchList( patchList, patchSize, factor ):
    (M,N) = patchList.shape

    # check that patch vector length is consistent w/ patchSize
    if np.prod( patchSize ) != M:
        print "upsamp - inconsistent patch size"
        print "patchList has ", M, "elements"
        print "patchSize arg has ", np.prod(patchSize), "elements"

        return None

    for i in range(0,N):
        patch = np.reshape( patchList[:,i], patchSize )
        patch_us = upsamplePatchZ( patch, factor )

        if i == 0:
            patchOut = np.zeros( [np.prod(patch_us.shape), N] )

        patchOut[:,i] = patch_us.flatten() 

    return patchOut

def upsamplePatchZ( patchIn, factor, kind='nearest' ):
    if not np.isscalar( factor ):
        zfactor = factor[2]

    half = (np.float32(zfactor) - 1) / 2
    x = np.arange(half, zfactor * (patchIn.shape[2]), zfactor)  
    xnew = np.arange( 0, zfactor * (patchIn.shape[2]))
    f = extrap1dNearest( interp1d( x, patchIn, kind ))

    # exptrap concatenates along the first dimension
    # so need to move the x-axis to the z-axis
    patchOut =  np.transpose( f(xnew), [ 1, 2, 0 ] )

    return patchOut

def extrap1dNearest(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[...,0]
        elif x > xs[-1]:
            return ys[...,-1]
        else:
            return interpolator(x)

    def ufunclike(xs):
        return np.array(map(pointwise, np.array(xs)))

    return ufunclike  

def extrap1dLinear(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return scipy.array(map(pointwise, scipy.array(xs)))

    return ufunclike  

def downsamplePatch3d( patchIn, factor, sourceSigmas, targetSigmas, kind='avg' ):
    if kind == 'gaussian':
        return downsamplePatch3dGauss( patchIn, factor, sourceSigmas, targetSigmas)
    else:
        return downsamplePatch3dAvgZ( patchIn, factor )


def downsamplePatch3dAvgZ( patchIn, factor ):
    if not np.isscalar( factor ):
        zfactor = factor[2]
    else:
        zfactor = factor

    sz = patchIn.shape
    patchOut = patchIn.reshape(-1,zfactor).mean(axis=1).reshape(sz[0],sz[1],sz[2]/zfactor)
    return patchOut


def downsamplePatch3dGauss( patchIn, factor, sourceSigmas, targetSigmas  ):
    tmp = factor * targetSigmas
    sig = np.sqrt( tmp * tmp - sourceSigmas * sourceSigmas )
    #print sig

    
    half = (np.float32(factor) - 1) / 2

    #patchOut = sp.gaussian_filter( patchIn, (0,0,0.8) ) 
    patchOut = sp.gaussian_filter( patchIn, sig ) 
    #print patchOut

    sz = patchIn.shape
    # Seem to need to add epsilon for expected rounding behavior at 0.5
    #newSz = np.round( (0.00001 + np.float32(sz)) / factor )
    #print "newSz",newSz

    x,y,z = np.meshgrid( np.arange(half[0], sz[0], step=factor[0]), 
                         np.arange(half[1], sz[1], step=factor[1]),
                         np.arange(half[2], sz[2], step=factor[2]))

    x = np.int32(np.round( x ))
    y = np.int32(np.round( y ))
    z = np.int32(np.round( z ))

    patchOut = patchIn[ x, y, z ] 
    
    return patchOut


