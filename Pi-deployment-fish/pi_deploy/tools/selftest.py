"""No hardware / weights / torch. Includes mock TCP and simulated engine lifecycle."""
import sys
import unittest
from pi_deploy import PROJECT_ROOT


def main():
    suite = unittest.defaultTestLoader.discover(str(PROJECT_ROOT / 'tests'), pattern='test_*.py',
                                              top_level_dir=str(PROJECT_ROOT))
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
