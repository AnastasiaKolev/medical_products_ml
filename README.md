# Курсовая работа: прогноз эффективности химических соединений

Данные: 1001 соединение, целевые показатели `IC50, mM`, `CC50, mM`, `SI` и числовые дескрипторы молекул.

## Структура

- `data/drug_activity.csv` - исходный датасет.
- `src/01_eda.py` - EDA и сохранение графиков/таблиц.
- `src/02_regression_ic50.py` - регрессия IC50.
- `src/03_regression_cc50.py` - регрессия CC50.
- `src/04_regression_si.py` - регрессия SI.
- `src/05_classification_ic50_median.py` - классификация IC50 > медианы.
- `src/06_classification_cc50_median.py` - классификация CC50 > медианы.
- `src/07_classification_si_median.py` - классификация SI > медианы.
- `src/08_classification_si_gt_8.py` - классификация SI > 8.
- `src/common.py` - общие функции.
- `src/run_all_experiments_fast.py` - запуск всех экспериментов.
- `results/` - метрики и таблицы.
- `reports/` - аналитический отчет и графики.

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python src/01_eda.py
PYTHONPATH=src python src/run_all_experiments_fast.py
```

## Важное методологическое решение

Для каждой задачи из признаков удаляются все три целевые колонки: `IC50, mM`, `CC50, mM`, `SI`. Это сделано для предотвращения утечки данных, так как `SI` рассчитывается на основе `IC50` и `CC50`.
