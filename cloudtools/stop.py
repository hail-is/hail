import argparse
from subprocess import check_call

def main(main_parser):

	parser = argparse.ArgumentParser(parents=[main_parser])
	args = parser.parse_args()

	check_call(['gcloud', 'dataproc', 'clusters', 'delete', args.name])
