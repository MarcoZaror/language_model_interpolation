from typing import List


def read_probabilities(file: str) -> List:
    with open(file, "r") as f:
        text = f.read()
        text = text.split("\n")
        text = [float(x) for x in text if x != ""]
    return text
