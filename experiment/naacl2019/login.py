#!/usr/bin/env /ihome/crc/wrappers/py_wrap.sh
''' crc-interactive.py -- An interactive Slurm helper
Usage:
    crc-interactive.py (-s | -g | -m) [-hv] [-t <time>] [-n <num-nodes>]
        [-p <partition>] [-c <num-cores>] [-u <num-gpus>] [-r <res-name>]
        [-b <memory>] [-a <account>] [-l <license>] [-f <feature>]

Positional Arguments:
    -s --smp                        Interactive job on smp cluster
    -g --gpu                        Interactive job on gpu cluster
    -m --mpi                        Interactive job on mpi cluster

Options:
    -h --help                       Print this screen and exit
    -v --version                    Print the version of crc-interactive.py
    -t --time <time>                Run time in hours, 1 <= time <= 12 [default: 1]
    -n --num-nodes <num-nodes>      Number of nodes [default: 1]
    -p --partition <partition>      Specify non-default partition
    -c --num-cores <num-cores>      Number of cores per node [default: 1]
    -u --num-gpus <num-gpus>        Used with -g only, number of GPUs [default: 1]
    -r --reservation <res-name>     Specify a reservation name
    -b --mem <memory>               Memory in GB
    -a --account <account>          Specify a non-default account
    -l --license <license>          Specify a license
    -f --feature <feature>          Specify a feature, e.g. `ti` for GPUs
'''


def check_integer_argument(arguments, key):
    if arguments[key]:
        try:
            return int(arguments[key])
        except ValueError:
            print("WARNING: {0} should have been an integer, setting {0} to 1 hr".format(key))
            return 1
    else:
        return 1


def add_to_srun_args(srun_args, srun_dict, arguments, item):
    if arguments[item]:
        srun_args += ' {} '.format(srun_dict[item]).format(arguments[item])
    return srun_args


def run_command(command):
    sp = Popen(split(command), stdout=PIPE, stderr=PIPE)
    return sp.communicate()

try:
    # Some imports functions and libraries
    from docopt import docopt
    from os import system
    from subprocess import Popen, PIPE
    from shlex import split

    arguments = docopt(__doc__, version='crc-interactive.py version 0.0.1')

    # Set defaults if necessary
    if not arguments['--time']:
        arguments['--time'] = 1
    if not arguments['--num-nodes']:
        arguments['--num-nodes'] = 1
    if not arguments['--num-cores']:
        arguments['--num-cores'] = 1
    if not arguments['--num-gpus']:
        arguments['--num-gpus'] = 1

    # Check the integer arguments
    arguments['--time'] = check_integer_argument(arguments, '--time')
    arguments['--num-nodes'] = check_integer_argument(arguments, '--num-nodes')
    arguments['--num-cores'] = check_integer_argument(arguments, '--num-cores')
    arguments['--num-gpus'] = check_integer_argument(arguments, '--num-gpus')
    arguments['--mem'] = check_integer_argument(arguments, '--mem')

    # Check walltime is between limits
    if not (arguments['--time'] >= 1 and arguments['--time'] <= 12):
        exit("ERROR: {} is not in 1 <= time <= 12... exiting".format(arguments['--time']))

    # Build up the srun arguments
    srun_dict = {'--partition': '--partition={}', '--num-nodes': '--nodes={}', '--num-cores': '--ntasks-per-node={}',
                 '--time': '--time={}:00:00', '--reservation': '--reservation={}', '--num-gpus': '--gres=gpu:{}',
                 '--mem': '--mem={}g', '--account': '--account={}', '--license': '--licenses={}',
                 '--feature': '--constraint={}'}
    srun_args = ""
    srun_args = add_to_srun_args(srun_args, srun_dict, arguments, '--partition')
    srun_args = add_to_srun_args(srun_args, srun_dict, arguments, '--num-nodes')
    srun_args = add_to_srun_args(srun_args, srun_dict, arguments, '--num-cores')
    srun_args = add_to_srun_args(srun_args, srun_dict, arguments, '--time')
    srun_args = add_to_srun_args(srun_args, srun_dict, arguments, '--reservation')
    srun_args = add_to_srun_args(srun_args, srun_dict, arguments, '--mem')
    srun_args = add_to_srun_args(srun_args, srun_dict, arguments, '--account')
    if arguments['--gpu']:
        srun_args = add_to_srun_args(srun_args, srun_dict, arguments, '--num-gpus')
    if arguments['--license']:
        srun_args = add_to_srun_args(srun_args, srun_dict, arguments, '--license')
    if arguments['--feature']:
        srun_args = add_to_srun_args(srun_args, srun_dict, arguments, '--feature')

    # Export MODULEPATH
    srun_args += ' --export=ALL '

    # Add --x11 flag?
    x11_out, x11_err = run_command("xset q")
    if len(x11_err) == 0:
        srun_args += ' --x11 '

    # Run the commands
    if arguments['--smp']:
        system("srun {} --pty bash".format(srun_args))
    elif arguments['--gpu']:
        system('ssh -Xt gpu-interactive "srun {} --pty bash --login"'.format(srun_args))
    elif arguments['--mpi']:
        if (not arguments['--partition'] == 'compbio') and arguments['--num-nodes'] < 2:
            exit('Error: You must use more than 1 node on the MPI cluster')
        system('ssh -Xt mpi-interactive "srun {} --pty bash --login"'.format(srun_args))
except KeyboardInterrupt:
    exit('Interrupt detected! exiting...')
