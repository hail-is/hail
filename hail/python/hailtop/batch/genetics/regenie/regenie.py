from hail.utils import hadoop_open as hopen
from hail.utils import hadoop_exists as hexists
from hailtop.batch import Resource, Batch, LocalBackend
from collections import namedtuple
import sys
import shlex
from argparse import Namespace, ArgumentParser
from typing import Set

BatchArgs = namedtuple("BatchArgs", ['cores', 'memory', 'storage'])
input_file_args = ["bgen", "bed", "pgen", "sample", "keep", "extract", "exclude", "remove",
                   "phenoFile", "covarFile"]

from_underscore = {
    "force_impute": "force-impute",
    "ignore_pred": "ignore-pred",
    "lowmem_prefix": "lowmem-prefix"
}


def _warn(msg):
    print(msg, file=sys.stderr)


def _error(msg):
    _warn(msg)
    sys.exit(1)


def add_shared_args(parser: ArgumentParser):
    # Batch knows in advance which step it is, so not required
    parser.add_argument('--step', required=False)
    parser.add_argument('--phenoFile', required=True)

    parser.add_argument('--phenoCol', required=False, action='append')
    parser.add_argument('--phenoColList', required=False)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--bed', required=False)
    group.add_argument('--bgen', required=False)
    group.add_argument('--pgen', required=False)

    parser.add_argument('--sample', required=False)
    parser.add_argument('--covarFile', required=False)
    parser.add_argument('--covarCol', required=False)
    parser.add_argument('--covarColList', required=False)
    parser.add_argument('--pThresh', required=False)
    parser.add_argument('--remove', required=False)
    parser.add_argument('--bsize', required=False)
    parser.add_argument('--cv', required=False)
    parser.add_argument('--nb', required=False)
    parser.add_argument('--out', required=False)

    parser.add_argument('--loocv', required=False, action='store_true')
    parser.add_argument('--bt', required=False, action='store_true')
    parser.add_argument('--1', '--cc12', required=False, action='store_true')
    parser.add_argument('--split', required=False, action='store_true')
    parser.add_argument('--strict', required=False, action='store_true')
    parser.add_argument('--firth', required=False, action='store_true')
    parser.add_argument('--approx', required=False, action='store_true')
    parser.add_argument('--spa', required=False, action='store_true')
    parser.add_argument('--debug', required=False, action='store_true')
    parser.add_argument('--verbose', required=False, action='store_true')
    parser.add_argument('--lowmem', required=False, action='store_true')

    parser.add_argument('--lowmem-prefix', required=False)
    parser.add_argument('--threads', required=False, default=2)


def add_step1_args(parser: ArgumentParser):
    parser.add_argument('--extract', required=False)
    parser.add_argument('--exclude', required=False)


def add_step2_args(parser: ArgumentParser):
    # Batch specifies private folders that regenie doesn't know, so --pred is ignored.
    parser.add_argument('--pred', required=False)
    parser.add_argument('--ignore-pred', required=False, action='store_true')

    parser.add_argument('--force-impute', required=False, action='store_true')
    parser.add_argument('--chr', required=False)


def read_step_args(path_or_str: str, step: int):
    parser = ArgumentParser()

    add_shared_args(parser)

    if step == 1:
        add_step1_args(parser)
    elif step == 2:
        add_step2_args(parser)
    else:
        _error(f"Unknown step: {step}")

    if not hexists(path_or_str):
        print(f"Couldn't find a file named {path_or_str}, assuming this is an argument string")
        r = parser.parse_args(shlex.split(path_or_str))
    else:
        print(f"Found {path_or_str}, reading")

        with hopen(path_or_str, "r") as f:
            t = shlex.split(f.read())
            r = parser.parse_args(t)

    if step == 2:
        if r.pred:
            print("Batch will set --pred to the output prefix of --step 1.")

    return r


def get_phenos(step_args: Namespace):
    phenos_to_keep = {}
    if step_args.phenoCol:
        for pheno in step_args.phenoCol:
            phenos_to_keep[pheno] = True

    if step_args.phenoColList:
        for pheno in step_args.phenoColList.split(","):
            phenos_to_keep[pheno] = True

    with hopen(step_args.phenoFile, "r") as f:
        phenos = f.readline().strip().split(" ")[2:]

    if not phenos_to_keep:
        return phenos

    phenos_final = []
    for pheno in phenos:
        if pheno in phenos_to_keep:
            phenos_final.append(pheno)

    return phenos_final


def prepare_step_cmd(batch: Batch, step_args: Namespace, job_output: Resource, skip: Set[str] = None):
    cmd = []
    for name, val in vars(step_args).items():
        if val is None or val is False or (skip is not None and name in skip):
            continue

        name = from_underscore.get(name, name)

        if name in input_file_args:
            if name == "bed":
                res = batch.read_input_group(bed=f"{val}.bed", bim=f"{val}.bim", fam=f"{val}.fam")
            elif name == "pgen":
                res = batch.read_input_group(pgen=f"{val}.pgen", pvar=f"{val}.pvar", psam=f"{val}.psam")
            else:
                res = batch.read_input(val)

            cmd.append(f"--{name} {res}")
        elif name == "out":
            cmd.append(f"--{name} {job_output}")
        elif isinstance(val, bool):
            cmd.append(f"--{name}")
        elif name == "phenoCol":
            for pheno in val:
                cmd.append(f"--{name} {pheno}")
        else:
            cmd.append(f"--{name} {val}")

    return ' '.join(cmd).strip()


def prepare_jobs(batch, args: BatchArgs, step1_args: Namespace, step2_args: Namespace):
    regenie_img = 'hailgenetics/regenie:9e7074f695e2b96bbb5af9d95f112d674c3260cd'
    j1 = batch.new_job(name='run-regenie-step1')
    j1.image(regenie_img)
    j1.cpu(args.cores)
    j1.memory(args.memory)
    j1.storage(args.storage)

    phenos = get_phenos(step1_args)
    nphenos = len(phenos)

    step1_output_prefix = step1_args.out
    s1out = {"log": f"{step1_output_prefix}.log", "pred_list": f"{step1_output_prefix}_pred.list"}

    for i in range(1, nphenos + 1):
        s1out[f"{step1_output_prefix}_{i}"] = f"{step1_output_prefix}_{i}.loco"

    if step1_args.lowmem:
        for i in range(1, nphenos + 1):
            pfile = f"{step1_args.lowmem_prefix}_l0_Y{i}"
            s1out[pfile] = pfile

    j1.declare_resource_group(**{step1_output_prefix: s1out})

    cmd1 = prepare_step_cmd(batch, step1_args, j1[step1_output_prefix])
    j1.command(f"regenie {cmd1}")

    phenos = get_phenos(step2_args)
    nphenos = len(phenos)

    j2 = batch.new_job(name='run-regenie-step2')
    j2.image(regenie_img)
    j2.cpu(args.cores)
    j2.memory(args.memory)
    j2.storage(args.storage)

    step2_output_prefix = step2_args.out
    s2out = {"log": f"{step2_output_prefix}.log"}

    if step2_args.split:
        for pheno in phenos:
            out = f"{step2_output_prefix}_{pheno}.regenie"
            s2out[out] = out

    print(f"Regenie Step 2 output files: \n{s2out.values()}")

    j2.declare_resource_group(**{step2_output_prefix: s2out})

    cmd2 = prepare_step_cmd(batch, step2_args, j2[step2_output_prefix], skip=set(['pred']))

    if not step2_args.ignore_pred:
        cmd2 = (f"{cmd2} --pred {j1[step1_output_prefix]['pred_list']}")

    j2.command(f"regenie {cmd2}")

    return j1, j2, step2_output_prefix


def run(args):
    is_local = args.local or args.demo

    if not is_local:
        _error("Currently only support LocalBackend (--local)")

    backend = LocalBackend()
    run_opts = {}

    if args.demo:
        if args.step1 or args.step2:
            _warn("When --demo provided, --step1 and --step2 are ignored")

        step1_args = read_step_args("example/step1.txt", 1)
        step2_args = read_step_args("example/step2.txt", 2)
    else:
        if not(args.step1 and args.step2):
            _error("When --demo not provided, --step1 and --step2 must be")

        step1_args = read_step_args(args.step1, 1)
        step2_args = read_step_args(args.step2, 2)

    batch_args = BatchArgs(cores=args.cores, memory=args.memory, storage=args.storage)

    batch = Batch(backend=backend, name='regenie')

    _, j2, j2_out_key = prepare_jobs(batch, batch_args, step1_args, step2_args)

    batch.write_output(j2[j2_out_key], args.out)
    batch.run(**run_opts)


def parse_input_args(args: list):
    parser = ArgumentParser()
    parser.add_argument('--local', required=False, action="store_true")
    parser.add_argument('--demo', required=False, action="store_true")
    parser.add_argument('--out', required=True)
    # FIXME: replace with per-step resources
    parser.add_argument('--cores', required=False, default=2)
    parser.add_argument('--memory', required=False, default="7Gi")
    parser.add_argument('--storage', required=False, default="1Gi")
    parser.add_argument('--step1', required=False)
    parser.add_argument('--step2', required=False)

    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_input_args(sys.argv[1:])
    run(args)
