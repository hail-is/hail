import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--tests', '-t',
                        type=str,
                        required=False,
                        help='Run specific comma-delimited tests instead of running all tests.')
    parser.add_argument('--cores', '-c',
                        type=int,
                        default=1,
                        help='Number of cores to allocate to Spark.')
    parser.add_argument("--n-iter", "-n",
                        type=int,
                        default=3,
                        help='Number of iterations for each test.')
    parser.add_argument("--log", "-l",
                         type=str,
                         help='Log file path')

    args = parser.parse_args()

    import benchmark
    benchmark.initialize(args.cores, args.log, args.n_iter)
    if args.tests:
        for test in args.tests.split(','):
            benchmark.run_single(test)
    else:
        benchmark.run_all()