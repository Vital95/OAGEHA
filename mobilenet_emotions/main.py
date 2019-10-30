from helper import *
import os
if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    args = parse_args()
    if not vars(args):
        ap.print_usage()
    else:
        args.func(args)