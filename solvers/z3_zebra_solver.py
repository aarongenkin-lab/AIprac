"""
Z3-based Zebra Logic Puzzle Solver
Uses SMT solver for constraint satisfaction
"""

import re
from typing import Dict, List, Tuple, Any, Optional
from z3 import *


class Z3ZebraSolver:
    """
    Solves Zebra logic puzzles using Z3 SMT solver.
    Converts natural language clues into Z3 constraints.
    """

    def __init__(self):
        self.solver = Solver()
        self.variables: Dict[str, Any] = {}
        self.categories: Dict[str, List[str]] = {}
        self.num_positions = 0

    def setup_puzzle(
        self,
        num_positions: int,
        categories: Dict[str, List[str]]
    ):
        """
        Setup the puzzle structure.

        Args:
            num_positions: Number of houses/positions
            categories: Dict mapping category name to list of values
                       e.g., {"person": ["Arnold", "Eric"], "car": ["Ford", "Tesla"]}
        """
        self.num_positions = num_positions
        self.categories = categories
        self.solver.reset()
        self.variables = {}

        # Create variables for each category and position
        for category, values in categories.items():
            for i in range(num_positions):
                var_name = f"{category}_{i}"
                var = Int(var_name)
                self.variables[var_name] = var

                # Domain constraint: value must be in valid range
                self.solver.add(And(var >= 0, var < len(values)))

            # Uniqueness constraint: each value appears exactly once
            position_vars = [self.variables[f"{category}_{i}"] for i in range(num_positions)]
            self.solver.add(Distinct(position_vars))

    def add_constraint_at_position(self, category: str, value: str, position: int):
        """
        Add constraint that a specific value is at a specific position.

        Example: Person "Arnold" is at position 1
        """
        value_idx = self.categories[category].index(value)
        var = self.variables[f"{category}_{position}"]
        self.solver.add(var == value_idx)

    def add_constraint_same_position(self, cat1: str, val1: str, cat2: str, val2: str):
        """
        Add constraint that two values are at the same position.

        Example: Person "Arnold" is in the same house as car "Tesla"
        """
        val1_idx = self.categories[cat1].index(val1)
        val2_idx = self.categories[cat2].index(val2)

        # Create disjunction: there exists a position where both are true
        constraints = []
        for i in range(self.num_positions):
            var1 = self.variables[f"{cat1}_{i}"]
            var2 = self.variables[f"{cat2}_{i}"]
            constraints.append(And(var1 == val1_idx, var2 == val2_idx))

        self.solver.add(Or(constraints))

    def add_constraint_adjacent(self, cat1: str, val1: str, cat2: str, val2: str, direction: str = "any"):
        """
        Add constraint that two values are in adjacent positions.

        Args:
            direction: "left" (val1 left of val2), "right", or "any"
        """
        val1_idx = self.categories[cat1].index(val1)
        val2_idx = self.categories[cat2].index(val2)

        constraints = []
        for i in range(self.num_positions):
            var1 = self.variables[f"{cat1}_{i}"]

            if direction in ["left", "any"] and i + 1 < self.num_positions:
                var2 = self.variables[f"{cat2}_{i+1}"]
                constraints.append(And(var1 == val1_idx, var2 == val2_idx))

            if direction in ["right", "any"] and i - 1 >= 0:
                var2 = self.variables[f"{cat2}_{i-1}"]
                constraints.append(And(var1 == val1_idx, var2 == val2_idx))

        if constraints:
            self.solver.add(Or(constraints))

    def add_constraint_directly_left(self, cat1: str, val1: str, cat2: str, val2: str):
        """Add constraint that val1 is directly to the left of val2"""
        self.add_constraint_adjacent(cat1, val1, cat2, val2, direction="left")

    def solve(self) -> Optional[Dict[int, Dict[str, str]]]:
        """
        Solve the puzzle.

        Returns:
            Dict mapping position to dict of category->value
            None if no solution exists
        """
        if self.solver.check() == sat:
            model = self.solver.model()
            solution = {}

            for i in range(self.num_positions):
                solution[i] = {}
                for category, values in self.categories.items():
                    var = self.variables[f"{category}_{i}"]
                    value_idx = model[var].as_long()
                    solution[i][category] = values[value_idx]

            return solution
        else:
            return None

    def print_solution(self, solution: Optional[Dict[int, Dict[str, str]]]):
        """Pretty print the solution"""
        if solution is None:
            print("No solution found!")
            return

        print("\nSOLUTION:")
        print("="*60)

        for pos in sorted(solution.keys()):
            print(f"\nPosition {pos + 1}:")
            for category, value in sorted(solution[pos].items()):
                print(f"  {category.capitalize()}: {value}")

        print("="*60)


def demo_simple_puzzle():
    """
    Demo: Solve the 2x3 puzzle from the user's example.

    Setup:
    - 2 Houses, 2 People (Arnold, Eric), 2 Cars (Ford F150, Tesla Model 3), 2 Animals (Cat, Horse)

    Clues:
    1. Eric is directly to the left of the person who owns a Tesla Model 3
    2. The person who keeps horses is in the first house

    Question: Who lives in House 2, and what car do they drive?
    """
    print("="*60)
    print("DEMO: 2x3 Zebra Logic Puzzle")
    print("="*60)

    solver = Z3ZebraSolver()

    # Setup puzzle structure
    categories = {
        "person": ["Arnold", "Eric"],
        "car": ["Ford F150", "Tesla Model 3"],
        "animal": ["Cat", "Horse"]
    }

    solver.setup_puzzle(num_positions=2, categories=categories)

    # Add clues
    print("\nClues:")
    print("1. Eric is directly to the left of the person who owns a Tesla Model 3")
    solver.add_constraint_directly_left("person", "Eric", "car", "Tesla Model 3")

    print("2. The person who keeps horses is in the first house")
    solver.add_constraint_at_position("animal", "Horse", 0)

    # Solve
    print("\nSolving...")
    solution = solver.solve()
    solver.print_solution(solution)

    # Answer the question
    if solution:
        print("\nANSWER:")
        house_2 = solution[1]
        print(f"House 2: {house_2['person']} drives the {house_2['car']}")


def demo_classic_puzzle():
    """
    Demo: A classic 5-house Zebra puzzle variation.

    Setup:
    - 5 houses in a row
    - 5 nationalities: English, Spanish, Ukrainian, Norwegian, Japanese
    - 5 colors: Red, Green, Ivory, Yellow, Blue
    - 5 pets: Dog, Snails, Fox, Horse, Zebra
    - 5 drinks: Coffee, Tea, Milk, Orange juice, Water

    Clues:
    1. The English person lives in the red house
    2. The Spanish person owns the dog
    3. Coffee is drunk in the green house
    4. The Ukrainian drinks tea
    5. The green house is immediately to the right of the ivory house
    6. The person who owns snails drinks orange juice
    7. The person in the yellow house drinks water
    8. Milk is drunk in the middle house
    9. The Norwegian lives in the first house
    10. The person who owns the fox lives next to the house with the horse
    """
    print("\n" + "="*60)
    print("DEMO: Classic 5-House Zebra Puzzle")
    print("="*60)

    solver = Z3ZebraSolver()

    categories = {
        "nationality": ["English", "Spanish", "Ukrainian", "Norwegian", "Japanese"],
        "color": ["Red", "Green", "Ivory", "Yellow", "Blue"],
        "pet": ["Dog", "Snails", "Fox", "Horse", "Zebra"],
        "drink": ["Coffee", "Tea", "Milk", "Orange juice", "Water"]
    }

    solver.setup_puzzle(num_positions=5, categories=categories)

    print("\nAdding clues...")

    # Clue 1: English in red house
    solver.add_constraint_same_position("nationality", "English", "color", "Red")

    # Clue 2: Spanish owns dog
    solver.add_constraint_same_position("nationality", "Spanish", "pet", "Dog")

    # Clue 3: Coffee in green house
    solver.add_constraint_same_position("drink", "Coffee", "color", "Green")

    # Clue 4: Ukrainian drinks tea
    solver.add_constraint_same_position("nationality", "Ukrainian", "drink", "Tea")

    # Clue 5: Green house immediately right of ivory
    solver.add_constraint_directly_left("color", "Ivory", "color", "Green")

    # Clue 6: Snail owner drinks orange juice
    solver.add_constraint_same_position("pet", "Snails", "drink", "Orange juice")

    # Clue 7: Yellow house has water
    solver.add_constraint_same_position("color", "Yellow", "drink", "Water")

    # Clue 8: Milk in middle house (position 2, 0-indexed)
    solver.add_constraint_at_position("drink", "Milk", 2)

    # Clue 9: Norwegian in first house
    solver.add_constraint_at_position("nationality", "Norwegian", 0)

    # Clue 10: Fox owner adjacent to horse
    solver.add_constraint_adjacent("pet", "Fox", "pet", "Horse")

    print("Solving...")
    solution = solver.solve()
    solver.print_solution(solution)

    if solution:
        # Find who owns the zebra
        for pos, attrs in solution.items():
            if attrs["pet"] == "Zebra":
                print(f"\nANSWER: The {attrs['nationality']} owns the zebra!")


def main():
    """Run demos"""
    demo_simple_puzzle()
    print("\n" + "="*80 + "\n")
    demo_classic_puzzle()


if __name__ == "__main__":
    main()
