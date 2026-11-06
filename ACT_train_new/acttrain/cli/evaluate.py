"""独立重新评测，或直接将参数传入统一evaluator。"""
import sys
from acttrain.evaluation.runner import main as evaluate

def main():
    return evaluate(sys.argv[1:])

if __name__=='__main__':main()
