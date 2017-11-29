from __future__ import print_function
import sys, os, math

import argparse 
import time

import numpy as np
from numpy import float32, int32, uint8, dtype, genfromtxt
from os.path import join

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load PyGreentea
# Absolute path to where PyGreentea resides
#pygt_path = '/groups/saalfeld/home/bogovicj/unet-tut/caffe-janelia/PyGreentea'
#pygt_path = '/groups/saalfeld/home/bogovicj/dev/pygt/PyGreentea'
#sys.path.append( pygt_path )

def main( loss_file, out, grid=None, do_log=False ):
    print( loss_file )
    
    if out:
        out_file = out
    else:
        if( do_log ):
            out_file = loss_file.replace( '.csv', '-logy.png' )
        else:
            out_file = loss_file.replace( '.csv', '.png' )

    print( out_file )

    loss_dat = genfromtxt( loss_file, delimiter=',' )
    x = loss_dat[:,0]
    y = loss_dat[:,1]
    print( y.shape )
    if( do_log ):
        ylog = np.zeros( y.shape )
        ylog[ y > 0 ] = np.log( y[ y > 0 ])
        plt.bar(x, ylog)  
    else:
        plt.bar(x,y)

    plt.xlabel('Error')
    plt.ylabel('Count')
    if grid:
        plt.axis( np.fromstring(grid,sep=','))

    plt.savefig( out_file )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( '-i', '--input', help="The input csv")
    parser.add_argument( '--logy', action='store_true', help="do semilog y", default=None )
    parser.add_argument( '-o', '--output', help="Output name", default=None)
    parser.add_argument( '-g', '--grid', help="Grid", default=None)
    args = parser.parse_args()

    do_log = args.logy
    if( do_log ):
        do_log = True
    print( 'do log', do_log )
    main( args.input, args.output, grid=args.grid, do_log=do_log )
    
