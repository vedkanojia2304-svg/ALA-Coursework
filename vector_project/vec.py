
# This file includes steps 1, 2 and 3 (Steps.txt)

import math
import random


class Vec:
    """A simple class for representing mathematical vectors."""

    def __init__(self, elements):
        """Create a vector from a list of numbers."""
        self.elements = list(elements)

        # Check that every element is a number
        for x in self.elements:
            if not isinstance(x, (int, float)):
                raise TypeError("Vector elements must be numbers.")

    def __add__(self, other):
        """Add two vectors element by element."""

        if not isinstance(other, Vec):
            raise TypeError("Can only add another Vec.")

        if len(self) != len(other):
            raise ValueError("Vectors must have the same dimensions.")

        result = []

        for x, y in zip(self.elements, other.elements):
            result.append(x + y)

        return Vec(result)

    def __sub__(self, other):
        """Subtract two vectors element by element."""

        if not isinstance(other, Vec):
            raise TypeError("Can only subtract another Vec.")

        if len(self) != len(other):
            raise ValueError("Vectors must have the same dimensions.")

        result = []

        for x, y in zip(self.elements, other.elements):
            result.append(x - y)

        return Vec(result)

    def __rmul__(self, scalar):
        """Multiply every element of the vector by a scalar."""

        if not isinstance(scalar, (int, float)):
            raise TypeError("Scalar must be a number.")

        result = []

        for x in self.elements:
            result.append(x * scalar)

        return Vec(result)

    def __imul__(self, scalar):
        """Multiply the vector by a scalar in place."""

        if not isinstance(scalar, (int, float)):
            raise TypeError("Scalar must be a number.")

        for i in range(len(self.elements)):
            self.elements[i] *= scalar

        return self

    def __neg__(self):
        """Return the negative of the vector."""
        return Vec([-x for x in self.elements])

    def __radd__(self, other):
        """Allow vectors to work with Python's sum() function."""

        if other == 0:
            return self

        return self + other

    def __iadd__(self, other):
        """Add another vector to this vector in place."""

        result = self + other
        self.elements = result.elements

        return self

    def __len__(self):
        """Return the number of elements in the vector."""
        return len(self.elements)

    def __repr__(self):
        """Return a readable representation of the vector."""
        return repr(self.elements)

    @staticmethod
    def zeros(n):
        """Create a vector containing n zeros."""
        return Vec([0] * n)

    @staticmethod
    def ones(n):
        """Create a vector containing n ones."""
        return Vec([1] * n)

    @staticmethod
    def uniform(n):
        """Create a vector containing n random numbers between 0 and 1."""
        return Vec([random.random() for _ in range(n)])

    def norm(self):
        """Calculate the Euclidean (L2) norm of the vector."""

        total = 0

        for x in self.elements:
            total += x * x

        return math.sqrt(total)
