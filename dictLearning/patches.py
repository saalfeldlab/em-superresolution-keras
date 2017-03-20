import numpy as np
import itertools
import math

def normalize( im ):
    print( normalize )
    im_min = np.min( im )
    im_max = np.max( im )
    return ( im - im_min ) / ( im_max - im_min )

def getPatchesExp( img, patchSize, N=-1, samples_in_cols=True, order='F'):
    print "getting patches experimental"
    pNumel = np.prod(patchSize)
    nDim   = patchSize.shape[0]

    sz = img.shape 
    impr = patchify( img, patchSize ).reshape( -1, np.prod( patchSize ))
    totNumPatches = impr.shape[0]

    if N > 0 :
        print "grabbing ", N, " patches"
        i = np.random.random_integers( 0, totNumPatches, N )
        patchList = impr[i,...]
    else:
        patchList = impr
      
    if samples_in_cols:
        patchList = np.transpose( patchList )

    if order == 'F':
        patchList = np.asfortranarray( patchList )

    return patchList

def patchify( img, patch_shape):

    img = np.ascontiguousarray(img)  # won't make a copy if not needed

    X, Y, Z = img.shape
    x, y, z = patch_shape

    shape = ((X-x+1), (Y-y+1), (Z-z+1), x, y, z) # number of patches, patch_shape
    #shape = ( 5, 3 ) # number of patches, patch_shape

    # The right strides can be thought by:
    # 1) Thinking of `img` as a chunk of memory in C order
    # 2) Asking how many items through that chunk of memory are needed when indices
    #    i,j,k,l are incremented by one

    #strides = img.itemsize*np.array([Y, 1, Y, 1])
    strides = img.itemsize*np.array([Z*Y, Y, 1, Z*Y, Y, 1])

    return np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides) 

def getPatchesCoordsEven( img, patchSize, coords, samples_in_cols=True, order='F'):

    pNumel = np.prod(patchSize)
    N = coords.shape[1]

    if samples_in_cols:
        patchList = np.zeros( [pNumel, N], order=order)
    else:
        patchList = np.zeros( [N, pNumel], order=order)

    for i in range(N):
        x = coords[ 0, i ]
        y = coords[ 1, i ]
        z = coords[ 2, i ]

        thisPatch = img[    x : x + patchSize[0], 
                            y : y + patchSize[1], 
                            z : z + patchSize[2] ]

        if samples_in_cols:
            patchList[:,i] = thisPatch.flatten()
        else:
            patchList[i,:] = thisPatch.flatten()

    return patchList

def getPatchesEven( img, patchSize, N=-1, samples_in_cols=True, order='F'):

    pNumel = np.prod(patchSize)
    nDim   = patchSize.shape[0]

    sz = img.shape  
    print 'patches sz ', sz
    print 'patches numel ', pNumel

    if N > 0 :
        #print "grabbing ", N, " patches"
        startIdx = np.zeros( nDim ) 
        print 'sz ', sz
        endIdx   = sz - patchSize - 1

        #coords = np.zeros( [nDim, N] )
        coords = np.zeros( [nDim] )

        # Strictly speaking, this does sampling with replacement
        # but duplicate values are unlikely for N << numel(img)
        # TODO consider a sampling without replacement
        #for d in range( 0, nDim ):
        #   coords[d,:] = np.random.random_integers( startIdx[d], endIdx[d], N ) 

        if samples_in_cols:
            patchList = np.zeros( [pNumel, N], order=order)
        else:
            patchList = np.zeros( [N, pNumel], order=order)

        print patchList.shape
            
        #for i in range(0,N):
        i = 0
        while i < N:
            for d in range( 0, nDim ):
                coords = np.random.random_integers( startIdx[d], endIdx[d], 3 ) 

            #print "x ", coords[0]
            #print "y ", coords[1]
            #print "z ", coords[2]

            thisPatch = img[    coords[0] : coords[0]+patchSize[0], 
                                coords[1] : coords[1]+patchSize[1], 
	                        coords[2] : coords[2]+patchSize[2] ]

            # patch should be at least 3/4 nonzero
            if np.float32( np.count_nonzero( thisPatch  )) / thisPatch.size > 0.75:
                if samples_in_cols:
                    patchList[:,i] = thisPatch.flatten()
                else:
                    patchList[i,:] = thisPatch.flatten()

                i = i + 1

    else:
        #print "grabbing all patches"
        # find the number of elements
        subSz = np.array(sz) - 2 * pRad
        N = np.prod( subSz ) 

        if samples_in_cols:
            patchList = np.zeros( [pNumel, N], order=order)
        else:
            patchList = np.zeros( [N, pNumel], order=order)

        for i,(x,y,z) in enumerate( itertools.product( *map( xrange, subSz ))):
            thisPatch = img[ x : x+patchSize[0], 
                             y : y+patchSize[1], 
							 z : z+patchSize[2] ]

            if samples_in_cols:
                patchList[:,i] = thisPatch.flatten()
            else:
                patchList[i,:] = thisPatch.flatten()

    return patchList

def getPatches( img, patchSize, N=-1, samples_in_cols=True, order='F'):

    pNumel = np.prod(patchSize)
    pRad   = (patchSize-1)/2
    nDim   = patchSize.shape[0]

    sz = img.shape  
    print 'patches sz ', sz
    print 'patches numel ', pNumel

    if N > 0 :
        #print "grabbing ", N, " patches"
        startIdx = pRad 
        print 'sz ',sz
        print 'prad ',pRad
        endIdx   = sz - pRad - 1
        #coords = np.zeros( [nDim, N] )
        coords = np.zeros( [nDim] )

        # Strictly speaking, this does sampling with replacement
        # but duplicate values are unlikely for N << numel(img)
        # TODO consider a sampling without replacement
        #for d in range( 0, nDim ):
        #   coords[d,:] = np.random.random_integers( startIdx[d], endIdx[d], N ) 

        if samples_in_cols:
            patchList = np.zeros( [pNumel, N], order=order)
        else:
            patchList = np.zeros( [N, pNumel], order=order)

        print patchList.shape
            
        #for i in range(0,N):
        i = 0
        while i < N:
            for d in range( 0, nDim ):
                coords = np.random.random_integers( startIdx[d], endIdx[d], 3 ) 

            #print "x ", coords[0]
            #print "y ", coords[1]
            #print "z ", coords[2]

            thisPatch = img[ coords[0]-pRad[0] : coords[0]+pRad[0]+1, 
                             coords[1]-pRad[1] : coords[1]+pRad[1]+1, 
							 coords[2]-pRad[2] : coords[2]+pRad[2]+1 ]

            # patch should be at least 3/4 nonzero
            if np.float32( np.count_nonzero( thisPatch  )) / thisPatch.size > 0.75:
                if samples_in_cols:
                    patchList[:,i] = thisPatch.flatten()
                else:
                    patchList[i,:] = thisPatch.flatten()

                i = i + 1

    else:
        #print "grabbing all patches"
        # find the number of elements
        subSz = np.array(sz) - 2 * pRad
        N = np.prod( subSz ) 

        if samples_in_cols:
            patchList = np.zeros( [pNumel, N], order=order)
        else:
            patchList = np.zeros( [N, pNumel], order=order)

        for i,(x,y,z) in enumerate( itertools.product( *map( xrange, subSz ))):
            thisPatch = img[ x : x+patchSize[0], 
                             y : y+patchSize[1], 
							 z : z+patchSize[2] ]

            if samples_in_cols:
                patchList[:,i] = thisPatch.flatten()
            else:
                patchList[i,:] = thisPatch.flatten()

    return patchList

def jointEntropy( hiPatch, loPatch, downsampleFactor, bitDepth ):
    """ Joint entropy 
    """


def conditionalEntropy( loPatch, downsampleFactor, bitDepth ):
    """ Entropy (H)
    """ 
    lut,pmf = generateLowToHiResLutPmf( downsampleFactor, bitDepth )
   
    H = 0.0 # the conditional entropy

    for value in loPatch.flatten():
        print value
        H += pmf[ value ] * math.log( pmf[ value ] )


def patchHypotheses( loPatch, downsampleFactor ):
    """ Expect loPatch to be 3d 
    """
    print "gen patch hypotheses"
    
    
def generateLowToHiResLutPmf( D, bitDepth ):
    """ Generate a look-up-table that gives all possible high res
    configurations that could give rise to a given low-res value.

        For every level at the input bitDepth v, generates a list of 
        D values also at that bitDepth whose average is v.
    """
    print "generateLowToHiResLUT for ", D, " downsample factor"

    numLevels = 2**bitDepth
    print "num levels: ", numLevels    

    # Generate all D*numLevels combinations of values
    s = np.indices( tuple( numLevels * np.ones(D,)))
    t = np.reshape( s, ( s.shape[0], int(np.prod( s.shape[1:]))))
   
    # Get the lo res values
    lut = np.round( np.mean( t, axis=0 ))
    pmf = np.float32(np.bincount( np.int64(lut) )) / lut.shape[0]

    return lut,pmf

    
