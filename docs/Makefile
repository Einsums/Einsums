# Makefile for Sphinx documentation
#

# PYVER needs to be major.minor, just "3" doesn't work - it will result in
# issues with the amendments to PYTHONPATH and install paths (see DIST_VARS).

# Use explicit "version_info" indexing since make cannot handle colon characters, and
# evaluate it now to allow easier debugging when printing the variable

PYVER:=$(shell python3 -c 'from sys import version_info as v; print("{0}.{1}".format(v[0], v[1]))')
PYTHON = python$(PYVER)

# You can set these variables from the command line.
SPHINXOPTS    ?=
SPHINXBUILD   ?= LANG=C sphinx-build
PAPER         ?=
DOXYGEN       ?= doxygen
# For merging a documentation archive into a git checkout of numpy/doc
# Turn a tag like v1.18.0 into 1.18
# Use sed -n -e 's/patttern/match/p' to return a blank value if no match
TAG ?= $(shell git describe --tag | sed -n -e's,v\([1-9]\.[0-9]*\)\.[0-9].*,\1,p')

FILES=

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -WT --keep-going -d build/doctrees $(PAPEROPT_$(PAPER)) \
                  $(SPHINXOPTS) source

.PHONY: help clean html web htmlhelp latex changes linkcheck \
	dist dist-build gitwash-update version-check html-build latex-build \
	merge-doc show docenv

#------------------------------------------------------------------------------

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  clean     to remove generated doc files and start fresh"
	@echo "  docenv    make a virtual environment in which to build docs"
	@echo "  html      to make standalone HTML files"
	@echo "  htmlhelp  to make HTML files and a HTML help project"
	@echo "  latex     to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  changes   to make an overview over all changed/added/deprecated items"
	@echo "  linkcheck to check all external links for integrity"
	@echo "  dist PYVER=... to make a distribution-ready tree"
	@echo "  gitwash-update GITWASH=path/to/gitwash  update gitwash developer docs"
	@echo "  merge-doc TAG=... to clone numpy/doc and archive documentation into it"
	@echo "  show      to show the html output in a browser"

clean:
	-rm -rf build/*
	find . -name generated -type d -prune -exec rm -rf "{}" ";"

#------------------------------------------------------------------------------
# Automated generation of all documents
#------------------------------------------------------------------------------

# Build the current numpy version, and extract docs from it.
# We have to be careful of some issues:
#
# - Everything must be done using the same Python version
#

#SPHINXBUILD="LANG=C sphinx-build"
GITVER ?= $(shell (cd ..; set -o pipefail && git rev-parse HEAD 2>/dev/null | cut -c1-7) || echo Unknown)

dist: build/dist.tar.gz

build/dist.tar.gz: real-dist

real-dist: html-build
	cp -r build/html build/dist
	cd build/html && zip -9r ../dist/einsums-html.zip .
	cd build/dist && tar czf ../dist.tar.gz *
	chmod ug=rwX,o=rX -R build/dist
	find build/dist -type d -print0 | xargs -0r chmod g+s

#------------------------------------------------------------------------------
# Basic Sphinx generation rules for different formats
#------------------------------------------------------------------------------

generate: build/generate-stamp
build/generate-stamp: $(wildcard source/reference/*.rst)
	mkdir -p build
	touch build/generate-stamp

html: html-build
html-build: generate
	mkdir -p build/html build/doctrees
	$(PYTHON) preprocess.py
ifeq (, $(shell which $(DOXYGEN)))
	@echo "Unable to find 'Doxygen:$(DOXYGEN)', skip generating C/C++ API from comment blocks."
else
	$(DOXYGEN) build/doxygen/Doxyfile
endif
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) build/html $(FILES)
	$(PYTHON) postprocess.py html build/html/*.html
	@echo
	@echo "Build finished. The HTML pages are in build/html."

htmlhelp: generate
	mkdir -p build/htmlhelp build/doctrees
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) build/htmlhelp $(FILES)
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in build/htmlhelp."

htmlhelp-build: htmlhelp build/htmlhelp/numpy.chm
%.chm: %.hhp
	-hhc.exe $^

qthelp: generate
	mkdir -p build/qthelp build/doctrees
	$(SPHINXBUILD) -b qthelp $(ALLSPHINXOPTS) build/qthelp $(FILES)

latex: latex-build
latex-build: generate
	mkdir -p build/latex build/doctrees
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) build/latex $(FILES)
	$(PYTHON) postprocess.py tex build/latex/*.tex
	perl -pi -e 's/LATEXOPTS =/LATEXOPTS ?= --halt-on-error/' build/latex/Makefile
	@echo
	@echo "Build finished; the LaTeX files are in build/latex."
	@echo "Run \`make all-pdf' or \`make all-ps' in that directory to" \
	      "run these through (pdf)latex."

coverage: build version-check
	mkdir -p build/coverage build/doctrees
	$(SPHINXBUILD) -b coverage $(ALLSPHINXOPTS) build/coverage $(FILES)
	@echo "Coverage finished; see c.txt and python.txt in build/coverage"

changes: generate
	mkdir -p build/changes build/doctrees
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) build/changes $(FILES)
	@echo
	@echo "The overview file is in build/changes."

linkcheck: generate
	mkdir -p build/linkcheck build/doctrees
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) build/linkcheck $(FILES)
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in build/linkcheck/output.txt."
texinfo:
	$(SPHINXBUILD) -b texinfo $(ALLSPHINXOPTS) build/texinfo
	@echo
	@echo "Build finished. The Texinfo files are in build/texinfo."
	@echo "Run \`make' in that directory to run these through makeinfo" \
	      "(use \`make info' here to do that automatically)."

info:
	$(SPHINXBUILD) -b texinfo $(ALLSPHINXOPTS) build/texinfo
	@echo "Running Texinfo files through makeinfo..."
	make -C build/texinfo info
	@echo "makeinfo finished; the Info files are in build/texinfo."


show:
	@python -c "import webbrowser; webbrowser.open_new_tab('file://$(PWD)/build/html/index.html')"
