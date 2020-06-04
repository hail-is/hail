.PHONY: all

GENERATED_HTML = $(patsubst %.md,%.html,$(wildcard *.md))

all: $(GENERATED_HTML)

$(GENERATED_HTML): %.html: %.md %.xslt template.xslt
	pandoc -s $< \
	  -f markdown \
	  -t html \
	  --mathjax \
	  --highlight-style=pygments \
	  --columns 10000 \
	  | xsltproc -o $@ --html $*.xslt -

clean:
	rm -f $(GENERATED_HTML)
