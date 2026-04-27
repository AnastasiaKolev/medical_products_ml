"""Task 4: classification IC50 > median."""
from run_all_experiments_fast import eval_clf
from common import load_data

if __name__ == "__main__":
    result = eval_clf(load_data(), "IC50, mM", "median", "ic50_gt_median")
    print(result)
