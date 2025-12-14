# Common Mathematical Formulas and Problem-Solving Strategies

## Financial Mathematics

### Compound Interest
**Formula:**
```
A = P(1 + r/n)^(nt)
```
Where:
- A = Final amount
- P = Principal (initial investment)
- r = Annual interest rate (as decimal)
- n = Number of times interest compounds per year
- t = Time in years

**Example:**
$10,000 at 4.5% annual rate, compounded monthly, for 7 years:
```python
P = 10000
r = 0.045
n = 12
t = 7
A = P * (1 + r/n) ** (n * t)
# A ≈ $14,376.03
```

### Simple Interest
**Formula:**
```
I = P × r × t
A = P + I
```

### Present Value / Future Value
**Future Value:**
```
FV = PV × (1 + r)^t
```

**Present Value:**
```
PV = FV / (1 + r)^t
```

## Geometry and Measurement

### Area Formulas
- **Rectangle:** A = length × width
- **Triangle:** A = ½ × base × height
- **Circle:** A = πr²
- **Trapezoid:** A = ½(b₁ + b₂) × h

### Volume Formulas
- **Cube:** V = s³
- **Rectangular prism:** V = l × w × h
- **Cylinder:** V = πr²h
- **Sphere:** V = (4/3)πr³

## Algebra

### Quadratic Formula
For ax² + bx + c = 0:
```
x = (-b ± √(b² - 4ac)) / (2a)
```

### Exponent Rules
- a^m × a^n = a^(m+n)
- a^m / a^n = a^(m-n)
- (a^m)^n = a^(mn)
- a^0 = 1
- a^(-n) = 1/a^n

### Logarithm Rules
- log(xy) = log(x) + log(y)
- log(x/y) = log(x) - log(y)
- log(x^n) = n × log(x)
- log_b(b) = 1
- log_b(1) = 0

## Statistics and Probability

### Mean (Average)
```
μ = (Σx) / n
```

### Standard Deviation
```
σ = √(Σ(x - μ)² / n)
```

### Probability
```
P(A) = (Number of favorable outcomes) / (Total number of outcomes)
```

### Combinations and Permutations
**Permutations** (order matters):
```
P(n, r) = n! / (n - r)!
```

**Combinations** (order doesn't matter):
```
C(n, r) = n! / (r! × (n - r)!)
```

## Distance, Rate, Time

### Basic Formula
```
Distance = Rate × Time
d = r × t
```

**Variations:**
```
Rate = Distance / Time
Time = Distance / Rate
```

### Relative Speed
**Same direction:**
```
Relative speed = |Speed₁ - Speed₂|
```

**Opposite directions:**
```
Relative speed = Speed₁ + Speed₂
```

## Problem-Solving Strategies

### 1. Understand the Problem
- Read carefully
- Identify what's given
- Identify what's being asked
- Draw a diagram if applicable

### 2. Devise a Plan
- Choose appropriate formula
- Break into smaller steps
- Consider edge cases

### 3. Execute the Plan
- Show your work step-by-step
- Use appropriate units
- Check intermediate results

### 4. Verify the Answer
- Does it make sense?
- Check units
- Substitute back into original problem
- Consider if answer is reasonable

## Common Problem Types and Approaches

### Word Problems
1. Define variables for unknowns
2. Write equations based on relationships
3. Solve algebraically
4. Check answer in context

### Optimization Problems
1. Write function to optimize
2. Find critical points (derivative = 0)
3. Check endpoints
4. Verify max/min

### Rate Problems
1. Set up rate equation
2. Use consistent units
3. Solve for unknown
4. Verify units in answer

## Tips for Accuracy

✅ **Always:**
- Write down formulas before plugging in numbers
- Use parentheses to clarify order of operations
- Keep track of units throughout calculation
- Round only at the final answer (use full precision in intermediate steps)

❌ **Avoid:**
- Rounding too early
- Mixing units (e.g., months and years)
- Skipping steps
- Not checking if answer makes sense

## Using Python for Calculations

When precision matters, use Python:

```python
import math

# Compound interest with high precision
P = 10000
r = 0.045
n = 12
t = 7

A = P * (1 + r/n) ** (n * t)
print(f"Final amount: ${A:.2f}")

# Statistical calculations
import statistics
data = [23, 45, 67, 12, 89, 34]
mean = statistics.mean(data)
stdev = statistics.stdev(data)

# Complex formulas
result = math.sqrt(sum(x**2 for x in data)) / len(data)
```

---

*Remember: Mathematics is about precision and logical reasoning. Show your work!*
