@echo off
call venv\Scripts\activate
python scripts\kaggle_train.py --limit-per-dataset 10000
pause
