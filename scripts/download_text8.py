from __future__ import annotations

import pathlib
import urllib.request
import zipfile


TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"


def main() -> None:
    output_dir = pathlib.Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / "text8.zip"
    text_path = output_dir / "text8.txt"

    if text_path.exists():
        print(f"Dataset already exists at {text_path}")
        return

    print(f"Downloading {TEXT8_URL} -> {zip_path}")
    urllib.request.urlretrieve(TEXT8_URL, zip_path)

    print("Extracting text8.txt")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract("text8", output_dir)

    extracted = output_dir / "text8"
    extracted.rename(text_path)

    print(f"Done: {text_path}")


if __name__ == "__main__":
    main()
