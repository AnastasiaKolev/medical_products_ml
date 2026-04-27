"""Task 3: regression for SI."""
from run_all_experiments_fast import eval_reg
from common import load_data

if __name__ == "__main__":
    result = eval_reg(load_data(), "SI")
    print(result)
