import argparse
from importlib import import_module

modes=["glucose","glucose_insulin","bistable","limit_cycle","linear","nagumo","nlink"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", type=str, default=None, nargs="?", help="/".join(modes)
    )
    parser.add_argument(
        "--path", type=str, default="dataset", help="output path"
    )
    parser.add_argument(
        "--prefix", type=str, default="", help="output"
    )
    parser.add_argument(
        "--num", type=int, default=10000, help="#data"
    )
    parser.add_argument(
        "--train_num", type=int, default=9000, help="#train_data"
    )
    parser.add_argument(
            "--input_type_id", type=int, default=-1, help="0: no input, 1: standard input"
    )
    parser.add_argument(
            "--input_scale", type=float, default=1, help="scale of input signal"
    )
    parser.add_argument(
        "--without_normalization", action="store_true", help="without normalization"
    )
    parser.add_argument(
        "--T", type=float, default=-1, help="T"
    )
    parser.add_argument(
        "--dh", type=float, default=-1, help="dh"
    )
    parser.add_argument(
        "--nlink_n", type=int, default=1, help="[nlink] #links"
    )
    parser.add_argument(
        "--nlink_q0", action="store_true", help="[nlink] output q_0"
    )
    parser.add_argument(
        "--nlink_qs", action="store_true", help="[nlink] output all q"
    )
    parser.add_argument(
        "--nlink_q0u0", action="store_true", help="[nlink] output q_0 u_0"
    )
    parser.add_argument(
        "--nlink_u0", action="store_true", help="[nlink] output u_0"
    )
    parser.add_argument(
        "--linear_x0", action="store_true", help="[linear] output v"
    )
    parser.add_argument(
        "--linear_x1", action="store_true", help="[linear] output v"
    )
    parser.add_argument(
        "--nlink_old_input", action="store_true", help="[nlink] old sin wave inputs"
    )
    parser.add_argument(
        "--init_random", action="store_true", help="[linear] initialize x0 at random "
    )
    args = parser.parse_args()
    if args.mode in modes:
        mod = import_module("ddm.data."+args.mode+"")
        mod.generate_dataset(args)
    else:
        print("unknown mode:"+args.mode)

if __name__ == "__main__":
    main()

