@echo off
REM Quick Start Script for Merging Pipeline
REM Use this to run common experiments

echo.
echo ============================================================
echo Merging Pipeline - Quick Start
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.7+
    exit /b 1
)

REM Check if requirements are installed
echo Checking dependencies...
python -c "import sklearn, nltk, matplotlib" 2>nul
if errorlevel 1 (
    echo Installing requirements...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Error: Failed to install requirements
        exit /b 1
    )
)

echo.
echo Select an experiment to run:
echo.
echo 1. Baseline (TF-IDF + TF-IDF)
echo 2. LSA Query Expansion + LSA Ranking
echo 3. WordNet Query Expansion + ESA Ranking
echo 4. Query Reduction + LSA Ranking
echo 5. N-gram Retrieval + LSA Ranking
echo 6. Grid Search (all combinations)
echo 7. Custom (enter your own parameters)
echo 0. Exit
echo.

set /p choice="Enter your choice (0-7): "

if "%choice%"=="0" goto :end
if "%choice%"=="1" goto :baseline
if "%choice%"=="2" goto :lsa_lsa
if "%choice%"=="3" goto :wordnet_esa
if "%choice%"=="4" goto :reduce_lsa
if "%choice%"=="5" goto :ngram_lsa
if "%choice%"=="6" goto :grid
if "%choice%"=="7" goto :custom
goto :invalid

:baseline
echo Running: TF-IDF + TF-IDF (Baseline)
python main.py --block1-mode none --block2-retrieval tfidf --block3-ranking tfidf
goto :end

:lsa_lsa
echo Running: LSA Query Expansion + LSA Ranking
python main.py --block1-mode lsa --block2-retrieval tfidf --block3-ranking lsa
goto :end

:wordnet_esa
echo Running: WordNet Query Expansion + ESA Ranking
python main.py --block1-mode wordnet --block2-retrieval ngram --block3-ranking esa
goto :end

:reduce_lsa
echo Running: Query Reduction + LSA Ranking
python main.py --block1-reduce --block1-mode none --block3-ranking lsa
goto :end

:ngram_lsa
echo Running: N-gram Retrieval + LSA Ranking
python main.py --block2-retrieval ngram --block3-ranking lsa
goto :end

:grid
echo Running: Grid Search (all combinations)
echo This will run multiple experiments - this may take a while
set /p liveplot="Launch interactive plot window after completion? (yes/no) [yes]: "
if "%liveplot%"=="" set liveplot=yes
pause
if /I "%liveplot%"=="yes" (
    python run_experiments.py --mode grid --interactive-plot
) else (
    python run_experiments.py --mode grid
)
goto :end

:custom
echo.
set /p block1="Enter Block 1 mode (none/lsa/esa/wordnet/word2vec/tfidf) [none]: "
if "%block1%"=="" set block1=none
set /p reduce="Enable query reduction? (yes/no) [no]: "
set /p block2="Enter Block 2 retrieval (tfidf/ngram/local_bow) [tfidf]: "
if "%block2%"=="" set block2=tfidf
set /p block3="Enter Block 3 ranking (tfidf/lsa/esa) [tfidf]: "
if "%block3%"=="" set block3=tfidf

if "%reduce%"=="yes" (
    python main.py --block1-mode %block1% --block1-reduce --block2-retrieval %block2% --block3-ranking %block3%
) else (
    python main.py --block1-mode %block1% --block2-retrieval %block2% --block3-ranking %block3%
)
goto :end

:invalid
echo Invalid choice. Please try again.
goto :end

:end
echo.
echo ============================================================
echo Thank you for using Merging Pipeline!
echo Check the output/ directory for results.
echo ============================================================
