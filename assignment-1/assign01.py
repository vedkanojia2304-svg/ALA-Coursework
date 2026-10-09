import sys
import math
from pathlib import Path
from typing import Self

# Get the path to the vector_project folder
project_path = Path(__file__).resolve().parent.parent / "vector_project"

# Add the folder to Python's search path
sys.path.insert(0, str(project_path))

# Import the existing Vec class
from vec import Vec


class Assignment1(Vec):

    # Calculate the mean
    def mean(self) -> float:
        if len(self.elements) == 0:
            raise ValueError("Cannot calculate mean of an empty vector")

        return sum(self.elements) / len(self.elements)

    # Return the demeaned vector
    def demean(self) -> Self:
        average = self.mean()
        return Assignment1([x - average for x in self.elements])

    # Calculate standard deviation
    def std(self) -> float:
        if len(self.elements) == 0:
            raise ValueError(
                "Cannot calculate standard deviation of an empty vector"
            )

        demeaned = self.demean()
        squared_deviations = [x * x for x in demeaned.elements]

        return math.sqrt(
            sum(squared_deviations) / len(self.elements)
        )


if __name__ == "__main__":
    v1 = Assignment1([1, 5, 6])

    print(v1)
    print(v1.mean())
    print(v1.demean())
    print(v1.std())
