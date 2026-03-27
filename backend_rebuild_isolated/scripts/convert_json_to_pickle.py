import json
import pickle
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print("usage: convert_json_to_pickle.py <input.json> <output.pkl>", file=sys.stderr)
        sys.exit(2)

    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    model = json.loads(in_path.read_text(encoding="utf-8"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
