import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def patch_hydra_argparse():
    original = argparse.ArgumentParser._check_help

    def check_help(self, action):
        if action.help is not None and not isinstance(action.help, str):
            action.help = repr(action.help)
        return original(self, action)

    argparse.ArgumentParser._check_help = check_help
