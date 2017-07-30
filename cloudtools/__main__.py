import argparse
import start
import submit
import connect
import diagnose
import stop

def main():

	main_parser = argparse.ArgumentParser(add_help=False)
	main_parser.add_argument('name', type=str, help='User-supplied name of Dataproc cluster.')
	main_parser.add_argument('module', choices=['start', 'submit', 'connect', 'diagnose', 'stop'], help='cloudtools command.')
	args, unparsed = main_parser.parse_known_args()

	if args.module == 'start':
		print("Starting cluster '{}'...".format(args.name))
		start.main(main_parser)

	elif args.module == 'submit':
		print("Submitting to cluster '{}'...".format(args.name))
		submit.main(main_parser)

	elif args.module == 'connect':
		print("Connecting to cluster '{}'...".format(args.name))
		connect.main(main_parser)

	elif args.module == 'diagnose':
		print("Diagnosing cluster '{}'...".format(args.name))
		diagnose.main(main_parser)

	elif args.module == 'stop':
		print("Stopping cluster '{}'...".format(args.name))
		stop.main(main_parser)


if __name__ == '__main__':
	main()
