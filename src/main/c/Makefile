.PHONY: default

CC=cc
CLANG_OPTS=-O3 -march=native -g -dynamiclib -std=gnu99 ibs.c -current_version 1.0.1 -compatibility_version 1.0.1 -fvisibility=hidden

default: lib/lib/darwin/libibs.dylib

lib/lib/darwin/libibs.dylib: ibs.c
	mkdir -p lib/lib/darwin
	${CC} ${CLANG_OPTS} -o lib/lib/darwin/libibs.dylib

clean:
	rm -r lib
