"""Task 6: classification SI > median."""
from run_all_experiments_fast import eval_clf
from common import load_data

if __name__ == "__main__":
    result = eval_clf(load_data(), "SI", "median", "si_gt_median")
    print(result)
