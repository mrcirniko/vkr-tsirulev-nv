$xelatex = 'xelatex -synctex=1 -interaction=nonstopmode -shell-escape %O %S';
$pdflatex = $xelatex;
ensure_path('TEXINPUTS', '..//');
$clean_ext = "aux bbl blg idx ind lof lot out toc acn acr alg glg glo gls fls log fdb_latexmk snm synctex.gz xdy glo-abr run.xml bcf";
