"""
main.py

Single entry point for the Website URL Phishing Detection module.

Usage:
    python main.py <url>              Predict a single URL and print the result
    python main.py                    Interactive mode (prompts for a URL)
    python main.py --serve            Launch the FastAPI server (uvicorn) on port 8000
    python main.py --evaluate         Run models/evaluate.py on the held-out test split
"""

import sys

from models.predict import predict_url


def print_result(result: dict) -> None:
    print("\nPrediction Result")
    print("-" * 60)
    for key, value in result.items():
        print(f"{key:<28}: {value}")
    print("-" * 60)


def run_serve() -> None:
    import uvicorn

    print("Starting API server at http://127.0.0.1:8000  (docs at /docs)")
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=True)


def run_evaluate() -> None:
    import runpy

    runpy.run_module("models.evaluate", run_name="__main__")


def main() -> None:
    args = sys.argv[1:]

    if not args:
        url = input("Enter URL: ").strip()
        print_result(predict_url(url))
        return

    if args[0] == "--serve":
        run_serve()
        return

    if args[0] == "--evaluate":
        run_evaluate()
        return

    url = args[0]
    print_result(predict_url(url))


if __name__ == "__main__":
    main()