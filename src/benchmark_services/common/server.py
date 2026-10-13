"""Container/local HTTP service entry point; no GPU initialization at startup."""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", choices=("spacellava", "ibm"), default=os.environ.get("MODEL_SERVICE"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    if args.service not in {"spacellava", "ibm"}:
        parser.error("Set MODEL_SERVICE or --service to spacellava/ibm")
    import uvicorn
    from .http_app import create_service
    uvicorn.run(create_service(args.service), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
