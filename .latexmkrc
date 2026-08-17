# CUMCM 模板（cumcmthesis.cls）要求 XeLaTeX 编译。
# 这里把 latexmk 的 pdflatex 规则替换为 xelatex，
# 即使 latexmk 以 -pdf（pdflatex）模式被调用也能正确编译。
$pdflatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error -recorder %O %S';
$pdf_mode = 5;
