"""Entry point used to run processing in its own process: run_pipeline.py <project_dir>"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    from fieldstitch.pipeline import run
    run(sys.argv[1])
