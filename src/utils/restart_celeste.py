import os
import psutil
import signal
from celeste_env import CelesteEnv

from itertools import count


def find_path(name: str) -> str:
    """find path of first process with name 'name'"""
    ls: list = [] # many processes can have same name make list of them
    for p in psutil.process_iter(['name', 'pid']):
        if p.info['name'] == name:
            ls.append(psutil.Process(p.info['pid']).exe())
    return ls[0]

def restart_celeste(env: CelesteEnv):
    name = 'Celeste.exe'
    path = find_path(name)
    for p in psutil.process_iter(['name', 'pid']):
        if p.info['name'] == name:
            os.kill(p.info['pid'], signal.SIGTERM)

    os.startfile(path)

    # test succesfull startup
    timeout = 90
    for i in count():
        try:
            env.reset()
            break
        except:
            if i >= timeout:
                raise




