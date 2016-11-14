.PHONY: default

CLANG_OPTS=-O3 -march=native -g -dynamiclib -std=gnu99 ibs.c -current_version 1.0.1 -compatibility_version 1.0.1 -fvisibility=hidden

default: lib/lib/darwin/ibs.dylib

lib/lib/darwin/ibs.dylib: ibs.c
	mkdir -p lib/lib/darwin
	clang ${CLANG_OPTS} -o lib/lib/darwin/ibs.dylib

clean:
	rm -r lib
