import argparse
import models

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=models.availabel_models)
    args = parser.parse_args()
    model = models.get_model(args)
    model.chat()
