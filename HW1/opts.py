'''
Hyperparameters wrapped in argparse
This file contains most of tuanable parameters for this homework
You are asked to play around with them for Q3.1
It is recommended that you leave them as they are before getting to Q3.1

You can change the values by changing their default fields or by command-line
arguments. For example, "python main.py --filter-scales 2 5 --K 50"
'''

import argparse

def get_opts():
    parser = argparse.ArgumentParser(description='16-720 HW1: Scene Recognition')

    ## Paths
    parser.add_argument('--data-dir', type=str, default='../data',
                        help='data folder')
    parser.add_argument('--feat-dir', type=str, default='../feat',
                        help='feature folder')
    parser.add_argument('--out-dir', type=str, default='.',
                        help='output folder')

    ## Visual words (requires tuning)
    '''
    # Default parameters
    parser.add_argument('--filter-scales', nargs='+', type=float,
                        default=[1, 2],
                        help='a list of scales for all the filters')
    parser.add_argument('--K', type=int, default=10,
                        help='# of words')
    parser.add_argument('--alpha', type=int, default=25,
                        help='Using only a subset of alpha pixels in each image') 

    ## Recognition system (requires tuning)
    parser.add_argument('--L', type=int, default=1,
                        help='L + 1 is the number of layers in spatial pyramid matching (SPM)')
'''
    ## Additional options (add your own hyperparameters here)
    '''
    # Exp-1
    parser.add_argument('--filter-scales', nargs='+', type=float,
                        default=[1, 2, 4],#, 6, 8, 10, 12, 14, 16],
                        help='a list of scales for all the filters')
    parser.add_argument('--K', type=int, default=50,
                        help='# of words')
    parser.add_argument('--alpha', type=int, default=50,
                        help='Using only a subset of alpha pixels in each image')
    
    parser.add_argument('--L', type=int, default=2,
                        help='L + 1 is the number of layers in spatial pyramid matching (SPM)')'''
    
    '''
    # Exp-2
    parser.add_argument('--filter-scales', nargs='+', type=float,
                        default=[1, 2, 4, 6],#, 8, 10, 12, 14, 16],
                        help='a list of scales for all the filters')
    parser.add_argument('--K', type=int, default=75,
                        help='# of words')
    parser.add_argument('--alpha', type=int, default=75,
                        help='Using only a subset of alpha pixels in each image')
    
    parser.add_argument('--L', type=int, default=3,
                        help='L + 1 is the number of layers in spatial pyramid matching (SPM)')
    '''

    '''
    # Exp-3
    parser.add_argument('--filter-scales', nargs='+', type=float,
                        default=[1, 2, 4, 6, 8],#, 12, 14, 16],
                        help='a list of scales for all the filters')
    parser.add_argument('--K', type=int, default=100,
                        help='# of words')
    parser.add_argument('--alpha', type=int, default=100,
                        help='Using only a subset of alpha pixels in each image')
    
    parser.add_argument('--L', type=int, default=3,
                        help='L + 1 is the number of layers in spatial pyramid matching (SPM)')
    '''
    '''
    # Exp-4
    parser.add_argument('--filter-scales', nargs='+', type=float,
                        default=[1, 2, 4, 6, 8, 10],#, 12, 14, 16],
                        help='a list of scales for all the filters')
    parser.add_argument('--K', type=int, default=100,
                        help='# of words')
    parser.add_argument('--alpha', type=int, default=100,
                        help='Using only a subset of alpha pixels in each image')
    
    parser.add_argument('--L', type=int, default=4,
                        help='L + 1 is the number of layers in spatial pyramid matching (SPM)')
    '''

    '''
    # Exp-5
    parser.add_argument('--filter-scales', nargs='+', type=float,
                        default=[1, 2, 4, 6, 8, 10],#, 12, 14, 16],
                        help='a list of scales for all the filters')
    parser.add_argument('--K', type=int, default=100,
                        help='# of words')
    parser.add_argument('--alpha', type=int, default=100,
                        help='Using only a subset of alpha pixels in each image')
    
    parser.add_argument('--L', type=int, default=3,
                        help='L + 1 is the number of layers in spatial pyramid matching (SPM)')
    '''
    '''
    # Exp-6
    parser.add_argument('--filter-scales', nargs='+', type=float,
                        default=[1, 2, 4, 6, 8],#, 10, 12, 14, 16],
                        help='a list of scales for all the filters')
    parser.add_argument('--K', type=int, default=100,
                        help='# of words')
    parser.add_argument('--alpha', type=int, default=100,
                        help='Using only a subset of alpha pixels in each image')
    
    parser.add_argument('--L', type=int, default=3,
                        help='L + 1 is the number of layers in spatial pyramid matching (SPM)')
    '''

    parser.add_argument('--filter-scales', nargs='+', type=float,
                        default=[1, 2, 4],#, 12, 14, 16],
                        help='a list of scales for all the filters')
    parser.add_argument('--K', type=int, default=10,
                        help='# of words')
    parser.add_argument('--alpha', type=int, default=25,
                        help='Using only a subset of alpha pixels in each image')
    
    parser.add_argument('--L', type=int, default=1,
                        help='L + 1 is the number of layers in spatial pyramid matching (SPM)')
    ##
    opts = parser.parse_args()
    return opts
