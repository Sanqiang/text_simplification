import os
import psutil
import subprocess


def print_cpu_memory():
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
    print('memory use:', memoryUse)


def print_gpu_memory():
    pipe = subprocess.Popen('nvidia-smi', stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    mteval_result = pipe.communicate()
    print(mteval_result[0].decode('utf-8'))