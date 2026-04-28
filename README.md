# Курсовая работа: прогноз эффективности химических соединений

Репозиторий оформлен в формате Jupyter Notebook.

## Состав

- `01_EDA.ipynb` — исследовательский анализ данных.
- `02_regression_IC50.ipynb` — регрессия для IC50.
- `03_regression_CC50.ipynb` — регрессия для CC50.
- `04_regression_SI.ipynb` — регрессия для SI.
- `05_classification_IC50_gt_median.ipynb` — классификация IC50 > медианы.
- `06_classification_CC50_gt_median.ipynb` — классификация CC50 > медианы.
- `07_classification_SI_gt_median.ipynb` — классификация SI > медианы.
- `08_classification_SI_gt_8.ipynb` — классификация SI > 8.
- `09_summary.ipynb` — сводная таблица лучших моделей.
- `data/drug_activity.csv` — исходные данные.
- `reports/coursework_report.docx` и `reports/coursework_report.pdf` — аналитический отчёт.

## Запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
jupyter notebook
```

Откройте ноутбуки и выполните ячейки сверху вниз.
