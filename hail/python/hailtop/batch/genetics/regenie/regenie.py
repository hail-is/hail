from ... import Batch, LocalBackend, ServiceBackend
from ...resource import Resource
import sys
import shlex
from argparse import Namespace, ArgumentParser, SUPPRESS
from typing import Set, Dict
from os.path import exists
from google.cloud import storage  # type: ignore
from google.cloud.storage.blob import Blob  # type: ignore


input_file_args = ["bgen", "bed", "pgen", "sample", "keep", "extract", "exclude", "remove",
                   "phenoFile", "covarFile"]

from_underscore = {
    "force_impute": "force-impute",
    "ignore_pred": "ignore-pred",
    "lowmem_prefix": "lowmem-prefix"
}


def _is_local(spath: str):
    if spath.startswith("gs://"):
        return False
    return True


def _read(spath: str):
    if _is_local(spath):
        with open(spath, "r") as f:
            return f.read()

    client = storage.Client()
    blob = Blob.from_string(spath, client)
    return blob.download_as_string().decode("utf-8")


def _read_first_line(spath: str):
    if _is_local(spath):
        with open(spath, "r") as f:
            return f.readline()
    return _read(spath).split("\n")[0]


def _exists(spath: str) -> bool:
    if _is_local(spath):
        return exists(spath)

    client = storage.Client()
    blob = Blob.from_string(spath, client)
    return blob.exists()


def _warn(msg):
    print(msg, file=sys.stderr)


def _error(msg):
    _warn(msg)
    sys.exit(1)


def add_shared_args(parser: ArgumentParser):
    # Batch knows in advance which step it is, so not required
    parser.add_argument('--step', required=False)

    parser.add_argument('--phenoFile', required=True)
    parser.add_argument('--out', required=True)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--bed', required=False)
    group.add_argument('--bgen', required=False)
    group.add_argument('--pgen', required=False)

    parser.add_argument('--phenoCol', required=False, action='append')
    parser.add_argument('--phenoColList', required=False)

    parser.add_argument('--sample', required=False)
    parser.add_argument('--covarFile', required=False)
    parser.add_argument('--covarCol', required=False)
    parser.add_argument('--covarColList', required=False)
    parser.add_argument('--pThresh', required=False)
    parser.add_argument('--remove', required=False)
    parser.add_argument('--bsize', required=False)
    parser.add_argument('--cv', required=False)
    parser.add_argument('--nb', required=False)

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


def add_step1_args(parser: ArgumentParser):
    parser.add_argument('--extract', required=False)
    parser.add_argument('--exclude', required=False)


def add_step2_args(parser: ArgumentParser):
    # Pred is derived from step 1, whenever step 1 is provided
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

    if not _exists(path_or_str):
        print(f"Couldn't find a file named {path_or_str}, assuming this is an argument string")
        t = shlex.split(path_or_str)
    else:
        print(f"Found {path_or_str}, reading")
        t = shlex.split(_read(path_or_str))

    regenie_args = parser.parse_known_args(t)[0]

    if step == 2:
        if regenie_args.pred:
            print("Batch will set --pred to the output prefix of --step 1.")

    bparser = ArgumentParser()
    bparser.add_argument('--threads', required=False, default=1)
    bparser.add_argument('--memory', required=False, default='1Gi')
    bparser.add_argument('--storage', required=False, default='1Gi')

    batch_args = bparser.parse_known_args(t)[0]

    return regenie_args, batch_args


def get_phenos(step_args: Namespace):
    phenos_to_keep = {}
    if step_args.phenoCol:
        for pheno in step_args.phenoCol:
            phenos_to_keep[pheno] = True

    if step_args.phenoColList:
        for pheno in step_args.phenoColList.split(","):
            phenos_to_keep[pheno] = True

    phenos = _read_first_line(step_args.phenoFile).strip().split(" ")[2:]

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
                res: Resource = batch.read_input_group(bed=f"{val}.bed", bim=f"{val}.bim", fam=f"{val}.fam")
            elif name == "pgen":
                res = batch.read_input_group(
                    pgen=f"{val}.pgen", pvar=f"{val}.pvar", psam=f"{val}.psam")
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


def prepare_jobs(batch, step1_args: Namespace, step1_batch_args: Namespace, step2_args: Namespace,
                 step2_batch_args: Namespace):
    regenie_img = 'hailgenetics/regenie:v1.0.5.6'
    j1 = batch.new_job(name='run-regenie-step1')
    j1.image(regenie_img)
    j1.cpu(step1_batch_args.threads)
    j1.memory(step1_batch_args.memory)
    j1.storage(step1_batch_args.storage)

    phenos = get_phenos(step1_args)
    nphenos = len(phenos)

    s1out = {"log": "{root}.log", "pred_list": "{root}_pred.list"}

    for i in range(1, nphenos + 1):
        s1out[f"pheno_{i}"] = f"{{root}}_{i}.loco"

    j1.declare_resource_group(output=s1out)
    cmd1 = prepare_step_cmd(batch, step1_args, j1.output)
    j1.command(f"regenie {cmd1}")

    phenos = get_phenos(step2_args)
    nphenos = len(phenos)

    j2 = batch.new_job(name='run-regenie-step2')
    j2.image(regenie_img)
    j2.cpu(step2_batch_args.threads)
    j2.memory(step2_batch_args.memory)
    j2.storage(step2_batch_args.storage)

    s2out = {"log": "{root}.log"}

    if step2_args.split:
        for pheno in phenos:
            out = f"{{root}}_{pheno}.regenie"
            s2out[f"{pheno}.regenie"] = out
    else:
        s2out["regenie"] = "{root}.regenie"

    j2.declare_resource_group(output=s2out)

    cmd2 = prepare_step_cmd(batch, step2_args, j2.output, skip=set(['pred']))

    if not step2_args.ignore_pred:
        cmd2 = (f"{cmd2} --pred {j1.output['pred_list']}")

    j2.command(f"regenie {cmd2}")

    return j2


def run(args: Namespace, backend_opts: Dict[str, any], run_opts: Dict[str, any]):
    is_local = "local" in args or "demo" in args

    if is_local:
        backend = LocalBackend(**backend_opts)
    else:
        backend = ServiceBackend(**backend_opts)

    has_steps = "step1" in args or "step2" in args
    if "demo" in args:
        if has_steps:
            _warn("When --demo provided, --step1 and --step2 are ignored")

        step1_args, step1_batch_args = read_step_args("example/step1.txt", 1)
        step2_args, step2_batch_args = read_step_args("example/step2.txt", 2)
    else:
        if not has_steps:
            _error("When --demo not provided, --step1 and --step2 must be")

        step1_args, step1_batch_args = read_step_args(args.step1, 1)
        step2_args, step2_batch_args = read_step_args(args.step2, 2)

    batch = Batch(backend=backend, name='regenie')

    j2 = prepare_jobs(batch, step1_args, step1_batch_args, step2_args, step2_batch_args)
    print(f"Will write output to: {step2_args.out}")
    batch.write_output(j2.output, step2_args.out)
    return batch.run(**run_opts)


def parse_input_args(input_args: list):
    parser = ArgumentParser(argument_default=SUPPRESS, add_help=False)
    parser.add_argument('--local', required=False, action="store_true",
                        help="Use LocalBackend instead of the default ServiceBackend")
    parser.add_argument('--demo', required=False, action="store_true",
                        help="Run Regenie using Batch LocalBackend and example/step1.txt, example/step2.txt step files")
    parser.add_argument('--step1', required=False,
                        help="Path to newline-separated text file of Regenie step1 arguments")
    parser.add_argument('--step2', required=False,
                        help="Path to newline-separated text file of Regenie step2 arguments")
    args = parser.parse_known_args(input_args)

    backend_parser = ArgumentParser(argument_default=SUPPRESS, add_help=False)
    if "local" in args[0] or "demo" in args[0]:
        backend_parser.add_argument('--tmp_dir', required=False,
                                    help="Batch LocalBackend `tmp_dir` option")
        backend_parser.add_argument('--gsa_key_file', required=False,
                                    help="Batch LocalBackend `gsa_key_file` option")
        backend_parser.add_argument('--extra_docker_run_flags', required=False,
                                    help="Batch LocalBackend `extra_docker_run_flags` option")

        run_parser = ArgumentParser(argument_default=SUPPRESS, parents=[parser, backend_parser], add_help=True,
                                    epilog="Batch LocalBackend options shown, try without '--local' to see ServiceBackend options")
        run_parser.add_argument('--dry_run', required=False, action="store_true",
                                help="Batch.run() LocalBackend `dry_run` option")
        run_parser.add_argument('--verbose', required=False, action="store_true",
                                help="Batch.run() LocalBackend `verbose` option")
        run_parser.add_argument('--delete_scratch_on_exit', required=False, action="store_true",
                                help="Batch.run() LocalBackend `delete_scratch_on_exit` option")
    else:
        backend_parser.add_argument('--billing_project', required=False,
                                    help="Batch ServiceBackend `billing_project` option")
        backend_parser.add_argument('--bucket', required=False,
                                    help="Batch ServiceBackend `bucket` option")

        run_parser = ArgumentParser(argument_default=SUPPRESS, parents=[parser, backend_parser], add_help=True,
                                    epilog="Batch ServiceBackend options shown, try '--local' to see LocalBackend options")
        run_parser.add_argument('--dry_run', required=False, action="store_true",
                                help="Batch.run() ServiceBackend  `dry_run` option")
        run_parser.add_argument('--verbose', required=False, action="store_true",
                                help="Batch.run() ServiceBackend `verbose` option")
        run_parser.add_argument('--delete_scratch_on_exit', required=False, action="store_true",
                                help="Batch.run() ServiceBackend `delete_scratch_on_exit` option")
        run_parser.add_argument('--wait', required=False, action="store_true",
                                help="Batch.run() ServiceBackend `wait` option")
        run_parser.add_argument('--open', required=False, action="store_true",
                                help="Batch.run() ServiceBackend `open` option")
        run_parser.add_argument('--disable_progress_bar', required=False, action="store_true",
                                help="Batch.run() ServiceBackend `disable_progress_bar` option")
        run_parser.add_argument('--callback', required=False,
                                help="Batch.run() ServiceBackend `callback` option")

    backend_args = backend_parser.parse_known_args(args[1])
    run_args = run_parser.parse_known_args(backend_args[1])

    return {"args": args[0], "backend_opts": vars(backend_args[0]), "run_opts": vars(run_args[0])}


if __name__ == '__main__':
    args = parse_input_args(sys.argv[1:])
    run(**args)
