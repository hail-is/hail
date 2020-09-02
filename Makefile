.DEFAULT_GOAL := default

default:
	echo Do not use this makefile to build hail, for information on how to \
	     build hail see: https://hail.is/docs/0.2/
	exit 1

check:
	make -C auth check
	make -C batch check
	make -C benchmark/python check
	make -C ci check
	make -C gear check
	make -C hail/python check
	make -C notebook check
	make -C query check
	make -C router-resolver check
	make -C scorecard check
	make -C web_common check
