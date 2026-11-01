"""Launch GaussianAvatar's official train.py unchanged, with one setting the CLI cannot express:

  white_background = False  (black background; arguments.py declares `_white_background = True` with
  action="store_true", so it can never be switched off from the command line).

The AvatarTest-v1 GT is image x mask on black and the renders must be on black, so training uses the same
background (see NOTES.md). Usage:  python ga_train.py <repo> <train.py arguments...>
"""
import os
import runpy
import sys


def main():
    repo = os.path.abspath(sys.argv[1])
    os.chdir(repo)                      # train.py / arguments.py resolve assets and smpl_files from os.getcwd()
    sys.path.insert(0, repo)
    import arguments
    sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
    import ga_compat
    ga_compat.apply()                   # CPU/CUDA index fix for pytorch 2.0.1, see ga_compat.py

    orig_extract = arguments.ModelParams.extract

    def extract(self, args):
        g = orig_extract(self, args)
        g.white_background = False
        return g

    arguments.ModelParams.extract = extract
    argv = sys.argv[2:]
    if "--train_stage" in argv and argv[argv.index("--train_stage") + 1] == "2":
        # train.py's TensorBoard logging calls scale_loss.item(), a variable that only exists in stage 1, so with
        # tensorboard installed (it is in environment.yml) stage 2 dies with UnboundLocalError at the first
        # iteration. Make its `from torch.utils.tensorboard import SummaryWriter` fail -> TENSORBOARD_FOUND = False
        # (train.py's own fallback: "not logging progress"). Logging only; training unchanged.
        sys.modules["torch.utils.tensorboard"] = None
    sys.argv = [os.path.join(repo, "train.py")] + sys.argv[2:]
    runpy.run_path(os.path.join(repo, "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
