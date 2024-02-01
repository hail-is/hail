.PHONY: doctest
doctest:
	pytest \
        -Werror:::batch -Werror:::hailtop -Werror::ResourceWarning \
	-v \
	-r A \
	--doctest-modules \
	--ignore=docs/conf.py \
	--doctest-glob='*.rst' \
        --doctest-continue-on-failure
