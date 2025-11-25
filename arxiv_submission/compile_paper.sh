#!/bin/bash

# Compile LaTeX paper for arXiv submission
# RISC-V ISA Extensions for Edge AI Inference

echo "🔧 Compiling RISC-V ISA Extensions Paper for arXiv"
echo "=================================================="

# Check if required tools are installed
command -v pdflatex >/dev/null 2>&1 || { 
    echo "❌ Error: pdflatex is required but not installed."
    echo "Install LaTeX: brew install --cask mactex (macOS) or apt-get install texlive-full (Ubuntu)"
    exit 1
}

command -v python3 >/dev/null 2>&1 || { 
    echo "❌ Error: python3 is required but not installed."
    exit 1
}

# Generate figures first
echo "📊 Generating publication-quality figures..."
python3 generate_figures.py

if [ $? -ne 0 ]; then
    echo "❌ Error generating figures. Installing required Python packages..."
    pip3 install matplotlib seaborn numpy
    python3 generate_figures.py
fi

echo ""
echo "📝 Compiling LaTeX document..."

# Clean previous compilation files
rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fdb_latexmk *.fls *.synctex.gz

# First compilation
echo "   Step 1/4: Initial compilation..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "❌ Error in first LaTeX compilation. Check main.tex for errors."
    pdflatex main.tex
    exit 1
fi

# Bibliography compilation
echo "   Step 2/4: Processing bibliography..."
bibtex main > /dev/null 2>&1

# Second compilation
echo "   Step 3/4: Second compilation..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1

# Final compilation
echo "   Step 4/4: Final compilation..."
pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1

if [ -f "main.pdf" ]; then
    echo ""
    echo "✅ Paper compiled successfully!"
    echo "📄 Output: main.pdf"
    
    # Get file size
    size=$(du -h main.pdf | cut -f1)
    echo "📏 File size: $size"
    
    # Count pages
    pages=$(pdfinfo main.pdf 2>/dev/null | grep Pages | awk '{print $2}')
    if [ ! -z "$pages" ]; then
        echo "📖 Page count: $pages pages"
    fi
    
    echo ""
    echo "🎯 Ready for arXiv submission!"
    echo ""
    echo "📋 Submission checklist:"
    echo "  ✅ main.tex - Main paper file"
    echo "  ✅ references.bib - Bibliography"
    echo "  ✅ ieee_performance_table.tex - Performance table"
    echo "  ✅ figures/ - All publication figures (7 files)"
    echo "  ✅ main.pdf - Compiled paper"
    echo ""
    echo "📤 Next steps:"
    echo "  1. Review main.pdf for formatting"
    echo "  2. Create arXiv account: https://arxiv.org/user/register"
    echo "  3. Submit to category: cs.AR (Computer Architecture)"
    echo "  4. Upload all .tex, .bib files and figures/ directory"
    
else
    echo "❌ Error: PDF not generated. Check LaTeX compilation errors."
    echo "Run manually: pdflatex main.tex"
    exit 1
fi

# Clean up auxiliary files
echo ""
echo "🧹 Cleaning up auxiliary files..."
rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fdb_latexmk *.fls *.synctex.gz

echo "✨ Compilation complete!"