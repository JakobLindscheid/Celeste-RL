import os
import psutil
import signal
from celeste_env import CelesteEnv

from itertools import count
from ahk import AHK


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

    # resize window
    ahk = AHK()

    win = ahk.win_get(title='ahk_exe Celeste.exe')

    if win:
        win.activate()
        win.to_top()

        try: # this will fail when run a second time
            win.set_style("-0xC40000")
        except:
            pass

        win.move(x=env.config.region[0], y=env.config.region[1], width=env.config.region[2], height=env.config.region[3])
        win.redraw()
    else:
        raise Exception("Celeste window not found")




