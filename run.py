import subprocess
import argparse


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, required=True,
                        help='Overrides config file')
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['train', 'test', 'visualize_attention'],
                        help='Choose which file to run')
    parser.add_argument('-n', '--nproc_per_node', type=int, default=4, required=True,
                        help='number of gpus per node')
    args = parser.parse_args()
    return args


def read_file_into_cli(fpath):
    """Read cli from file into a string."""
    res = []
    with open(fpath, 'r') as fin:
        for line in fin:
            args = line.split('#')[0].strip()
            if len(args) == 0:
                continue
            res.append(args)
    return res


def escape_str(input_str):
    return f"'{input_str}'"


def construct_cmd(args):
    if args.cfg:
        assert args.cfg.startswith("expts"), "Must be wrt this directory"

    cli_stuff = read_file_into_cli(args.cfg)
    cli_stuff = [escape_str(el) for el in cli_stuff]
    cli_stuff = ' '.join(cli_stuff)

    cli = (f'HYDRA_FULL_ERROR=1 torchrun --nproc_per_node={args.nproc_per_node} {args.mode}.py ')
    cli += cli_stuff
    return cli


def main():
    args = parse_args()
    cmd = construct_cmd(args)
    print('>> Running "{}"'.format(cmd))
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    main()