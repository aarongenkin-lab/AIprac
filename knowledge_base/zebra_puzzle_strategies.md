# Zebra Puzzle Solving Strategies

Zebra puzzles (also called Einstein puzzles or logic grid puzzles) are constraint satisfaction problems that require systematic deductive reasoning.

## Overview
A typical zebra puzzle involves:
- **N positions** (usually 5 houses in a row)
- **Multiple attributes** per position (color, nationality, pet, drink, cigarette brand)
- **Clues** that constrain the relationships between attributes
- **Goal**: Determine which attributes go together at each position

## Step-by-Step Solution Strategy

### Step 1: Set Up Your Grid
Create a grid to track possibilities:
```
Position:  1    2    3    4    5
Color:
Nation:
Pet:
Drink:
Cigarette:
```

Or use a constraint matrix to track what CAN'T be together.

### Step 2: Categorize Clues
Organize clues by type:

**Direct Assignment Clues:**
- "The Englishman lives in the red house"
- "The Norwegian lives in the first house"
- Action: Immediately mark these in your grid

**Pair Clues:**
- "The Swede keeps dogs"
- "Kools are smoked in the yellow house"
- Action: Link these attributes (they occur at same position)

**Neighbor Clues:**
- "The green house is next to the white house"
- "The man who smokes Blends lives next to the cat owner"
- Action: Positions differ by exactly 1 (adjacent)

**Directional Clues:**
- "The green house is immediately to the right of the ivory house"
- Action: If ivory at position N, green at position N+1

**Relative Position Clues:**
- "The Norwegian lives next to the blue house"
- Action: These could be in either order (Norwegian at N, Blue at N±1)

### Step 3: Process Direct Facts First
1. Mark all direct assignments
2. Mark all definite pairs
3. Mark positions that are definitively NOT possible

Example:
- "Norwegian at position 1" → Mark (1, Norwegian) = TRUE
- "Green immediately right of Ivory" → Green cannot be at position 1

### Step 4: Apply Constraints Iteratively

**Use Process of Elimination:**
- Each attribute appears EXACTLY ONCE
- If 4 positions ruled out for an attribute, the 5th must have it
- If attribute X is at position 3, mark all other positions as NOT X

**Combine Clues:**
Example:
- Clue A: "Norwegian lives in first house"
- Clue B: "Norwegian lives next to blue house"
- Deduction: Blue house must be at position 2

**Chain Reasoning:**
Example:
- Clue 1: "Englishman lives in red house"
- Clue 2: "Red house is next to green house"
- Clue 3: "Green house owner drinks coffee"
- Chain: Englishman → Red → Next to Green → Coffee nearby

### Step 5: Handle Positional Clues Carefully

**"Next to" means adjacent (either direction):**
- If A next to B, and B at position 3, then A at position 2 OR 4
- Check both possibilities

**"Immediately to the right" is directional:**
- A to right of B, and B at 2, then A at 3 (not 1)
- Right means higher position number

**"To the left" is also directional:**
- A to left of B means position(A) < position(B)
- This doesn't specify adjacent

### Step 6: Use Logical Inference Patterns

**Pattern 1: Forced Placement**
If only one position possible for an attribute, place it there.

**Pattern 2: Mutual Exclusion**
If attribute X is at position N, no other position can have X.

**Pattern 3: Pair Propagation**
If A and B always occur together, and A is at position N, then B is at position N.

**Pattern 4: Neighbor Chains**
If A next to B, and B next to C, explore possible orderings: ABC, CBA.

### Step 7: Systematic Iteration
1. Apply all constraints once
2. Mark new definite deductions
3. Check if any new eliminations are possible
4. Repeat until:
   - Puzzle solved, OR
   - No more deductions possible → may need hypothesis testing

### Step 8: Hypothesis Testing (If Stuck)
If pure deduction doesn't solve it:
1. Identify an uncertain cell with few possibilities (2-3 options)
2. Assume one value is true
3. Follow all implications
4. If contradiction found, that assumption was wrong
5. If consistent, continue with that assumption

## Common Mistakes to Avoid

❌ **Mistake 1:** Assuming "next to" means "immediately to the right"
✅ **Correct:** "Next to" means adjacent in EITHER direction

❌ **Mistake 2:** Forgetting uniqueness constraints
✅ **Correct:** Each attribute appears exactly once across all positions

❌ **Mistake 3:** Missing implications of clues
✅ **Correct:** If X at position 3, then X NOT at positions 1, 2, 4, 5

❌ **Mistake 4:** Mixing up "to the right" vs "next to"
✅ **Correct:** "To the right" is directional; "next to" is not

❌ **Mistake 5:** Not tracking negative information
✅ **Correct:** Keep a grid of what CAN'T be true, not just what must be true

## Tips for Efficiency

1. **Start with most restrictive clues** (direct assignments, specific positions)
2. **Look for attributes that appear in multiple clues** (they give more information)
3. **Use edge positions** (1 and 5) - they have fewer neighbors
4. **Check consistency frequently** - catch errors early
5. **Draw the grid visually** - humans are good at visual patterns

## Example Walkthrough

**Clues:**
1. The Englishman lives in the red house
2. The Norwegian lives in the first house
3. The Norwegian lives next to the blue house

**Solution Steps:**

Step 1: Process clue 2 (direct)
```
Position 1: Norwegian
```

Step 2: Combine clues 2 and 3
- Norwegian at position 1
- Norwegian next to blue
- Therefore: Blue house at position 2 (only neighbor of position 1)

Step 3: Process clue 1
- Englishman in red house
- We know: Position 1 = Norwegian (so NOT Englishman)
- We know: Position 2 = Blue (so NOT red)
- Therefore: Englishman in red house at position 3, 4, or 5

This demonstrates how combining clues yields new information!

## When to Use Programming

For complex puzzles (more than 5 positions or 6+ attributes), consider:
- **Constraint satisfaction solver** (backtracking algorithm)
- **Boolean satisfiability (SAT)** solver
- **Logic programming** (Prolog)

But understand the manual approach first—it develops intuition!

## Additional Resources

- Grid method: Draw a table and mark X for impossible, ✓ for confirmed
- Constraint propagation: Each deduction triggers more deductions
- Backtracking: Systematically try possibilities and undo if contradiction

---

*Remember: Every zebra puzzle is solvable using pure logic—no guessing required!*
