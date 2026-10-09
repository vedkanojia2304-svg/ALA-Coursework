
# Demonstration of the Vec class

from vec import Vec


# Create two vectors
v1 = Vec([1, 2, 3])
v2 = Vec([4, 5, 6])

print("Vector 1:", v1)
print("Vector 2:", v2)

# Vector addition
print("Addition:", v1 + v2)

# Vector subtraction
print("Subtraction:", v2 - v1)

# Scalar multiplication
print("Scalar multiplication:", 2 * v1)

# In-place multiplication
v3 = Vec([1, 2, 3])
v3 *= 3
print("In-place multiplication:", v3)

# Negation
print("Negation:", -v1)

# Zero vector
print("Zeros:", Vec.zeros(5))

# One vector
print("Ones:", Vec.ones(4))

# Random vector
print("Random vector:", Vec.uniform(3))

# Euclidean norm
v4 = Vec([3, 4])
print("Norm:", v4.norm())
