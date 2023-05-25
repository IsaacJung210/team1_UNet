import argparse

def read_train_arg():
    parser = argparse.ArgumentParser(description="Start Training U-net")
    parser.add_argument("load", help="T[true] or F[false]")
    parser.add_argument("-p", "--pth", help="Select *.pth file")

    return parser.parse_args()

def read_infer_arg():
    # mode = check mean of full time / check once time & write image
    parser = argparse.ArgumentParser(description="Start U-net Inference")
    parser.add_argument("pth", help="Select *.pth file")

    return parser.parse_args()