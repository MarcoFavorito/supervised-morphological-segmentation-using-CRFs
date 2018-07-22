import argparse

import demos.demo_manager
import log
import settings
from utils import print_log


def main():
    logger = log.setup_logger(__name__)
    print_log(logger, "Start main")

    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", type=int,
                        help="determine the demo to be executed (1|2|3)")
    parser.add_argument("--verbosity", help="increase output verbosity", action="store_true")
    args = parser.parse_args()

    if args.verbosity:
        settings.VERBOSE = True
        print_log(logger, "verbosity turned on")
    if args.demo:
        demo_id = args.demo
    else:
        parser.error("You have to decide one of the available demo (1|2|3)")
        return -1

    print_log(logger, "Selected demo: %d" % demo_id)

    demos.demo_manager.exec_demo(demo_id)

    print_log(logger, "Demo %s done." % demo_id)
    return 0


if __name__ == "__main__":
    main()
