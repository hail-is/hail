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

    args = parser.parse_args()

    import benchmark
    if args.tests:
        for test in args.tests.split(','):
            benchmark.run_single(test, cores=args.cores, n_iter=args.n_iter)
    else:
        benchmark.run_all(cores=args.cores, n_iter=args.n_iter)