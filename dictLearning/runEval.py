import sys
import h5py 
import numpy as np
import spams

from dictLearning import evaluation
from dictLearning import patches
import evaluation_metrics as metrics 

import argparse

#def normalize( im ):
#    print( 'normalizing' )
#    im_min = np.min( im )
#    im_max = np.max( im )
#    return ( im - im_min ) / ( im_max - im_min )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( '--image', '-i', required=True, type=str, 
          help='The source hdf5 image' )
    parser.add_argument( '--dictionary', '-d', default="", type=str, 
          help='The dictionary' )
    parser.add_argument( '--output', '-o', default="", type=str, 
          help='The output prediction' )
    parser.add_argument( '--factor', '-f', default=4, type=int, 
          help='The output prediction' )
    parser.add_argument( '--batch-size', '-b', default=2000, type=int, 
          help='The output prediction' )
    parser.add_argument( '--patch-size', '-p', default='8-8-8', type=str, 
          help='Patch size' )
    parser.add_argument( '--step-size', '-s', default='', type=str, 
          help='Step size' )
    parser.add_argument( '--image-internal-directory', default='main', type=str, 
          help='Internal directory of hdf5 image' )

    args = parser.parse_args()

    print( args.image )
    print( args.dictionary )
    print( args.batch_size )

    outf = ""
    if args.output:
        outf = args.output
    else:
        outf = (args.dictionary).replace( '.h5', '_pred.h5' )

    print( outf )

    dsfactor = args.factor
    patchSize = np.fromstring( args.patch_size, dtype=int, sep='-')
    print "patch size: ", patchSize
   
    stepSize = None
    if args.step_size:
        stepSize = np.fromstring( args.step_size, dtype=int, sep='-')
        print "step size: ", stepSize

    # load the image
    f = h5py.File( args.image )
    im = np.squeeze( f[ args.image_internal_directory ][...] )
    print "shape ", im.shape

    # load the dictionary
    dfn = h5py.File( args.dictionary )
    D = dfn[ 'dict' ][...]
    dparamsGrp = dfn[ 'param' ]

    # load the parameters
    lparam = {}
    lparam[ 'lambda1' ] = dparamsGrp[ 'lambda1' ][...][()]
    lparam[ 'mode' ] = dparamsGrp[ 'mode' ][...][()]
    lparam[ 'numThreads' ] = dparamsGrp[ 'numThreads' ][...][()]

    im_norm = patches.normalize( im )

    imup = evaluation.dictUpsampleImage( im_norm, D, lparam, dsfactor,
            patchSize, batchSize=args.batch_size, stepSize=stepSize )

    print( im_norm.shape )
    print( imup.shape )

    mse, psnr, wmse, wpsnr = metrics.run_eval( im_norm,  imup, axis=2 )

    imuph5 = h5py.File( outf, 'a' )
    imuph5.create_dataset( 'estimate', data = imup.astype('float'))
    imuph5.flush()
    imuph5.close()

