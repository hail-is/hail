.PHONY: doctest
doctest:
	pytest \
        -Werror:::batch -Werror:::hailtop \
	-v \
	-r A \
	--doctest-modules \
	--ignore=docs/conf.py \
	--doctest-glob='*.rst'
