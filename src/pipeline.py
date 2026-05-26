from . import data, features, train, evaluate, predict


def run_full_setup() -> dict:
    print("=== Step 1/3: Data ingestion ===")
    data.run()
    print("\n=== Step 2/3: Feature engineering ===")
    features.run()
    print("\n=== Step 3/3: Training ===")
    result = train.run()
    print("\n=== Setup complete ===")
    return result


def run_evaluation() -> dict:
    return evaluate.run()


def run_prediction(features: dict = None) -> float:
    return predict.run(features)
