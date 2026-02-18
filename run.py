import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, help="Pipeline stage")
    args = parser.parse_args()

    if args.stage == "acquire":
        print("Running data acquisition...")
    elif args.stage == "preprocess":
        print("Running preprocessing...")
    elif args.stage == "features":
        print("Building features...")
    elif args.stage == "train":
        print("Training model...")
    elif args.stage == "eval":
        print("Evaluating model...")
    else:
        print("Invalid stage")

if __name__ == "__main__":
    main()