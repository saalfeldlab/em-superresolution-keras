import os
import sys
import time
sys.path.append( os.path.dirname( os.path.realpath( __file__ ) ) )

from dictLearning import *
import numpy as np
import h5py
import spams

import evaluation_metrics as metrics

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( '--image', '-i', required=True, type=str, 
          help='The source hdf5 image' )
    parser.add_argument( '--test-image', '-s', default="", type=str, 
          help='The test hdf5 image' )
    parser.add_argument( '--downsample-factor', '-f', default=4, type=int, 
          help='Downsampling factor in z' )
    parser.add_argument( '--output', '-o', default="", type=str, 
          help='The output hdf5 file where the dictionary will be stored' )
    parser.add_argument( '--image-internal-directory', '-d', default='main', type=str, 
          help='Internal directory of hdf5 image' )
    parser.add_argument( '--number', '-n', default='-1', type=int, 
          help='Number of patches' )
    parser.add_argument( '--dictionary-size', '-k', default=100,
          type=int, help='Dictionary size' )
    parser.add_argument( '--lambda1', '-l', default=0.15,
          type=float, help='Lambda (sparsity parameter)' )
    parser.add_argument( '--mode', '-m', default=2,
          type=int, help='Optimization mode' )
    parser.add_argument( '--patch-size', '-p', default='8-8-8', type=str, 
          help='Patch size' )
    parser.add_argument( '--iterations', '-t', default=1000, type=int,
          help='Dictionary learning iterations' )
    parser.add_argument( '--batch-size', '-b', default=100, type=int, help="Batch size")

    parser.add_argument( '--objective-function', dest='do_objective', action='store_true',
          help="Compute and print value of objective function")
    parser.add_argument( '--normalize', action='store_true',
          help="Compute and print value of objective function")

    parser.add_argument( '--dictionary-in', default='', type=str, 
            help='A pre-learned dictionary file')

    parser.add_argument( '--threads', '-r', default=1, type=int, 
          help='Number of threads ' )
    parser.add_argument( '--verbose', '-v', dest='verbose', action='store_true',
          help='Verbose output' )

    args = parser.parse_args()
    K = args.dictionary_size
    N = args.number

    dict_in_fn = args.dictionary_in

    dsFactor = args.downsample_factor
    doUpTest = False 
    if dsFactor > 1:
        doUpTest = True 
        ds = np.array([1,1,dsFactor])

    patchSize = np.fromstring( args.patch_size, dtype=int, sep='-')
    print "patch size: ", patchSize

    print "reading data"
    print args.image , "/", args.image_internal_directory

    f = h5py.File( args.image )
    im = np.squeeze( f[ args.image_internal_directory ][...] )
    #im = np.squeeze( f[ args.image_internal_directory ] )
    print "shape ", im.shape

    print "sampling ", N, " patches"
    tic = time.time()


    X = patches.getPatchesEven( im, patchSize, N )
    toc = time.time()
    t = toc - tic
    print 'time to grab patches: %f' % t 


    if dict_in_fn:

        print "loading dictionary from file"

    
        dfn = h5py.File( dict_in_fn )	

        if 'params' in dfn.keys():
            # Read dictionary parameters
            dparamsGrp = dfn[ 'params']
            tmp = []
            for k in dparamsGrp.keys():
                tmp.append( (k, dparamsGrp[k]))    

            params = dict( tmp )
        else:
            params = {          'K' : K,
                             'mode' : args.mode,
                          'lambda1' : args.lambda1, 
                       'numThreads' : args.threads,
                        'batchsize' : args.batch_size,
                             'iter' : args.iterations,
                          'verbose' : args.verbose }


        tic = time.time()
        D = dfn[ 'dict'][...]
        toc = time.time()
        print 'time to load dictionary: %f' % t 

    else:
        params = {          'K' : K,
                         'mode' : args.mode,
                      'lambda1' : args.lambda1, 
                   'numThreads' : args.threads,
                    'batchsize' : args.batch_size,
                         'iter' : args.iterations,
                      'verbose' : args.verbose }


        print "learning dictionary"
        tic = time.time()
        D = spams.trainDL( X, **params )
        toc = time.time()
        t = toc - tic
        print 'time of computation for Dictionary Learning: %f' % t 


    ###############################
    lst = [ 'L','lambda1','lambda2','mode','pos','ols','numThreads','length_path','verbose','cholesky']
    lparam = {'return_reg_path' : False}
    for x in lst:
        if x in params:
            lparam[x] = params[x]
    ###############################

    R = None
    #if True:
    if args.do_objective: 
        print "computing objective function"


        tic = time.time()
        R = evaluation.dictEval( X, D, lparam )
        toc = time.time()
        t = toc - tic
        print " TRAIN objective function value: %f" % R
        print 'time of computation for objective function: %f' % t
    else:
        print "Skipping objective function"

    #if doUpTest:
    #    tic = time.time()
    #    #Xd,dsPatchSz = evaluation.downsamplePatchList( X, patchSize, ds)
    #    #Dd,tmp2 = evaluation.downsamplePatchList( D, patchSize, ds)
    #    Ru = evaluation.dictEval( X, D, lparam, lam=0, dsfactor=ds, patchSize=patchSize, patchFnGrp=paramGrpTrn)
    #    toc = time.time()
    #    t = toc - tic
    #    print " TRAIN objective function on downsampled value: %f" % Ru
    #    print 'time of computation for objective function: %f' % t

    #    # upsample test
    #    Ruu = evaluation.upsampEval( X, ds, patchSize, patchFnGrp=paramGrpTst, kind='avg' )
    #    print " objective function on downsampled NN upsampled value: %f" % Ruu


    # Write the dictionary
    if args.output:
        print "Writing dictionary"

        h5out = h5py.File( args.output, 'a' )
        h5out.create_dataset("dict", data=D)
        if R:
            h5out.create_dataset("objective", data=R)

        # save parameters
        paramGroup = h5out.create_group("param")
        for pKey in params.keys():
            paramGroup.create_dataset( pKey, data=params[pKey])

        # save image and number of samples
        paramGroup.create_dataset( 'source_img', data=args.image )
        paramGroup.create_dataset( 'numSamples', data=N )

        h5out.flush()
        h5out.close()


    if args.test_image:
        print( 'test image: ', args.test_image )
        # read the image
        ft = h5py.File( args.test_image )
        imt = np.squeeze( f[ args.image_internal_directory ][...] )

        tic = time.time()
        #Xtd,dsPatchSz = evaluation.downsamplePatchList( Xt, patchSize, ds)
        #Dd,tmp2 = evaluation.downsamplePatchList( D, patchSize, ds)
        #Rtu = evaluation.dictEval( Xt, D, lparam, lam=0, dsfactor=ds, patchSize=patchSize, patchFnGrp=paramGrpTst )

        usefulKeys = [ 'lambda1', 'mode', 'numThreads' ]
        lassoParams = { k : params[k] for k in usefulKeys }

        imt_norm = patches.normalize( imt ) 
        im_up = evaluation.dictUpsampleImage( imt_norm, D, lassoParams,
                dsFactor, patchSize, batchSize=5000, stepSize=[4,4,4] )

        toc = time.time()
        t = toc - tic
        print 'time of computation for upsampling: %f' % t

        print( im_up.shape )
        print( imt_norm.shape )

        mse, psnr, wmse, wpsnr = metrics.run_eval( imt_norm, im_up, axis=2 )

        imuph5 = h5py.File( outf, 'a' )
        imuph5.create_dataset( 'estimate', data=imup.astype('float'))
        imuph5.flush()
        imuph5.close()


    
    if patches_out:
        patchFn.close()
        
    sys.exit( 0 )
