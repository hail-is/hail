# requires OS X & brew

REVIEWERS := jon,danking,danielg,patrick,arcturus,jon,jackie,chris,tim,alex

build/poppler:
	mkdir -p build
	brew install poppler
	touch build/poppler

build/gs:
	mkdir -p build
	brew install gs
	touch build/gs

build/pdfseparate: build/poppler
	mkdir -p build
	touch build/pdfseparate

build/split: coops.pdf build/pdfseparate
	mkdir -p build
	mkdir -p build/split.d
	pdfseparate $< build/split.d/%d.pdf
	touch build/split

build/packets: build/split build/gs
	mkdir -p docs
	mkdir -p build/packets.d
	python3 doit.py $(shell ls build/split.d | wc -l) $(REVIEWERS)
	touch build/packets

clean:
	rm -rf build
