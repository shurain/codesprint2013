MAIN = main
TARGET = codesprint2013r2
LATEX = pdflatex

all: $(TARGET).article.pdf $(TARGET).beamer.pdf

$(TARGET).article.pdf: article.tex $(MAIN).tex
	$(LATEX) --jobname=$(TARGET).article article

$(TARGET).beamer.pdf: beamer.tex $(MAIN).tex
	$(LATEX) --jobname=$(TARGET).beamer beamer

link:
	@ln -s $(CURDIR)/$(TARGET).article.pdf ~/public_html/$(TARGET).article.pdf
	@ln -s $(CURDIR)/$(TARGET).beamer.pdf ~/public_html/$(TARGET).beamer.pdf

clean:
	@rm -f $(TARGET).article.aux $(TARGET).article.log $(TARGET).article.pdf $(TARGET).article.nav $(TARGET).article.out $(TARGET).article.snm $(TARGET).article.toc $(TARGET).article.vrb *~
	@rm -f $(TARGET).beamer.aux $(TARGET).beamer.log $(TARGET).beamer.pdf $(TARGET).beamer.nav $(TARGET).beamer.out $(TARGET).beamer.snm $(TARGET).beamer.toc $(TARGET).beamer.vrb *~
	@rm -f pygments.aux
