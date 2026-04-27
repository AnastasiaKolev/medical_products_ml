"""Task 7: classification SI > 8."""
from run_all_experiments_fast import eval_clf
from common import load_data

if __name__ == "__main__":
    result = eval_clf(load_data(), "SI", "si_gt_8", "si_gt_8")
    print(result)
