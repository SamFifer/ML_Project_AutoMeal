
import pandas as pd
import numpy as np
import ast
import re
import random

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = r"C:\Users\samjf\Downloads\ML LLM Project\ML_Project_AutoMeal\recipes.csv"
OUTPUT_FILE = "ML_Project_AutoMeal\recipes_10000_entrees.csv"

TARGET_SIZE = 10000
RANDOM_SEED = 42

# Minimum recipe quality requirements
MIN_INGREDIENTS = 2
MIN_INSTRUCTIONS = 2
MIN_INSTRUCTION_LENGTH = 15

# ============================================================
# 1. CATEGORY CLASSIFICATION
# ============================================================

# Categories that strongly indicate an entree or main meal.
# Matching is based on phrases, not just exact category names.

POSITIVE_CATEGORY_TERMS = [
    "main dish",
    "main dishes",
    "main course",
    "one dish meal",
    "chicken",
    "beef",
    "pork",
    "turkey",
    "lamb",
    "veal",
    "meat",
    "poultry",
    "seafood",
    "fish",
    "shrimp",
    "crab",
    "lobster",
    "pasta",
    "noodles",
    "rice",
    "beans",
    "lentils",
    "vegetarian",
    "vegan",
    "tofu",
    "casserole",
    "stew",
    "soup",
    "curry",
    "lasagna",
    "pizza",
    "sandwich",
    "burgers",
    "meatloaf",
    "roast",
    "stir fry",
    "quiche",
    "tacos",
    "enchiladas",
    "fajitas",
    "kebabs",
]

# These categories override positive matches.
NEGATIVE_CATEGORY_TERMS = [
    "dessert",
    "cookie",
    "cake",
    "pie",
    "candy",
    "chocolate",
    "ice cream",
    "frozen dessert",
    "beverage",
    "drink",
    "cocktail",
    "smoothie",
    "juice",
    "coffee",
    "tea",
    "snack",
    "appetizer",
    "side dish",
    "side dishes",
    "sauce",
    "dip",
    "spread",
    "condiment",
    "dressing",
    "marinade",
    "bread",
    "muffin",
    "jam",
    "jelly",
    "preserves",
    "pickle",
    "seasoning",
    "garnish",
    "breakfast",
    "brunch",
]

# Additional phrases that suggest a recipe is not an entree.
NEGATIVE_RECIPE_TERMS = [
    "dessert",
    "snack",
    "appetizer",
    "side dish",
    "beverage",
    "drink",
    "cocktail",
    "smoothie",
    "juice",
    "cookie",
    "brownie",
    "cupcake",
    "frosting",
    "icing",
    "candy",
    "ice cream",
    "sorbet",
    "pudding",
    "sweet treat",
    "trail mix",
    "party mix",
    "dip",
    "spread",
    "salsa",
    "sauce",
    "dressing",
    "marinade",
    "condiment",
    "bread",
    "muffin",
    "jam",
    "jelly",
    "pickle",
    "breakfast",
    "pancake",
    "waffle",
    "garlic butter",
    "seasoning mix",
]

# Positive terms in recipe names and keywords.
# These help identify entrees even when the category is vague.

POSITIVE_RECIPE_TERMS = [
    "main dish",
    "main course",
    "entree",
    "dinner",
    "lunch",
    "chicken",
    "beef",
    "pork",
    "turkey",
    "lamb",
    "steak",
    "meatloaf",
    "meatballs",
    "roast",
    "fish",
    "salmon",
    "tuna",
    "shrimp",
    "seafood",
    "pasta",
    "lasagna",
    "spaghetti",
    "noodles",
    "rice bowl",
    "fried rice",
    "casserole",
    "stew",
    "soup",
    "curry",
    "stir fry",
    "stir-fry",
    "tacos",
    "enchiladas",
    "fajitas",
    "burritos",
    "quesadillas",
    "pizza",
    "burger",
    "sandwich",
    "kebabs",
    "kebab",
    "quiche",
    "pot pie",
    "stuffed peppers",
    "tofu",
    "vegetarian entree",
    "vegan entree",
]

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def parse_list(value):
    """
    Convert Food.com R-style vectors and Python-style lists
    into Python lists.

    Examples:
        c("item 1", "item 2")
        ["item 1", "item 2"]
    """

    if isinstance(value, list):
        return value

    if not isinstance(value, str):
        return None

    value = value.strip()

    if not value:
        return None

    # Handle R-style vectors.
    if value.startswith("c(") and value.endswith(")"):
        inner = value[2:-1].strip()

        if not inner:
            return None

        try:
            result = ast.literal_eval("[" + inner + "]")

            if isinstance(result, list):
                return result

        except (ValueError, SyntaxError):
            return None

    # Handle normal Python lists.
    try:
        result = ast.literal_eval(value)

        if isinstance(result, list):
            return result

    except (ValueError, SyntaxError):
        pass

    return None


def normalize_text(value):
    """Normalize text for searching and duplicate detection."""

    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value


def contains_phrase(text, phrases):
    """Check for whole-word or whole-phrase matches."""

    if not isinstance(text, str):
        return False

    text = normalize_text(text)

    for phrase in phrases:
        phrase = normalize_text(phrase)

        pattern = r"\b" + re.escape(phrase) + r"\b"

        if re.search(pattern, text):
            return True

    return False


def valid_ingredients(value):
    """Check whether a recipe has enough usable ingredients."""

    if not isinstance(value, list):
        return False

    ingredients = [
        str(item).strip()
        for item in value
        if item is not None and str(item).strip()
        and str(item).strip().lower() not in ["nan", "na"]
    ]

    return len(ingredients) >= MIN_INGREDIENTS


def valid_quantities(ingredients, quantities):
    """
    Make sure every ingredient has a corresponding quantity.
    """

    if not isinstance(ingredients, list):
        return False

    if not isinstance(quantities, list):
        return False

    if len(ingredients) != len(quantities):
        return False

    for quantity in quantities:
        if quantity is None:
            return False

        quantity = str(quantity).strip().lower()

        if not quantity or quantity in ["nan", "na", "none"]:
            return False

    return True


def valid_instructions(value):
    """Check for a usable set of cooking instructions."""

    if not isinstance(value, list):
        return False

    instructions = [
        str(step).strip()
        for step in value
        if step is not None and str(step).strip()
    ]

    if len(instructions) < MIN_INSTRUCTIONS:
        return False

    total_length = sum(len(step) for step in instructions)

    return total_length >= MIN_INSTRUCTION_LENGTH


def is_entree(row):
    """
    Positive entree classification.

    A recipe qualifies if its category or its name/keywords
    provide positive evidence of being a main meal.

    Negative matches always override positive matches.
    """

    category = normalize_text(row["RecipeCategory"])
    name = normalize_text(row["Name"])

    keywords = normalize_text(row.get("Keywords", ""))

    combined_text = f"{name} {keywords}"

    # First reject explicitly non-entree categories.
    if contains_phrase(category, NEGATIVE_CATEGORY_TERMS):
        return False

    # Reject explicitly non-entree recipe names/keywords.
    if contains_phrase(combined_text, NEGATIVE_RECIPE_TERMS):
        return False

    # Check whether category indicates a main meal.
    category_match = contains_phrase(
        category,
        POSITIVE_CATEGORY_TERMS
    )

    # Check whether name or keywords indicate a main meal.
    recipe_match = contains_phrase(
        combined_text,
        POSITIVE_RECIPE_TERMS
    )

    return category_match or recipe_match


def ingredient_signature(ingredients):
    """
    Create a normalized ingredient signature to help
    identify duplicate recipes.
    """

    if not isinstance(ingredients, list):
        return ""

    normalized = sorted(
        normalize_text(item)
        for item in ingredients
        if item is not None
    )

    return "|".join(normalized)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 60)
print("FOOD.COM RECIPE FILTER")
print("=" * 60)

print("\nLoading CSV...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

# Clean column names.
df.columns = (
    df.columns
    .str.strip()
    .str.replace("\ufeff", "", regex=False)
)

original_count = len(df)

print(f"Original recipes: {original_count:,}")

# Make sure all required columns exist.
REQUIRED_COLUMNS = [
    "RecipeId",
    "Name",
    "RecipeCategory",
    "Keywords",
    "RecipeIngredientQuantities",
    "RecipeIngredientParts",
    "RecipeInstructions",
]

missing = [
    col for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

# ============================================================
# 4. PARSE RECIPE DATA
# ============================================================

print("\nParsing ingredient and instruction lists...")

LIST_COLUMNS = [
    "RecipeIngredientQuantities",
    "RecipeIngredientParts",
    "RecipeInstructions",
]

for col in LIST_COLUMNS:
    df[col] = df[col].apply(parse_list)

print("Parsing complete.")

# ============================================================
# 5. REMOVE INCOMPLETE RECIPES
# ============================================================

print("\nChecking recipe completeness...")

before_quality = len(df)

# Require an ID, name, and category.
df = df.dropna(
    subset=[
        "RecipeId",
        "Name",
        "RecipeCategory",
    ]
).copy()

# Remove empty names and categories.
df = df[
    df["Name"].astype(str).str.strip().ne("")
    & df["RecipeCategory"].astype(str).str.strip().ne("")
].copy()

# Require enough ingredients.
df = df[
    df["RecipeIngredientParts"].apply(valid_ingredients)
].copy()

# Require quantities matching the ingredients.
df = df[
    df.apply(
        lambda row: valid_quantities(
            row["RecipeIngredientParts"],
            row["RecipeIngredientQuantities"]
        ),
        axis=1
    )
].copy()

# Require usable cooking instructions.
df = df[
    df["RecipeInstructions"].apply(valid_instructions)
].copy()

quality_count = len(df)

print(f"Recipes passing quality checks: {quality_count:,}")
print(f"Removed incomplete recipes: {before_quality - quality_count:,}")

# ============================================================
# 6. POSITIVE ENTREE FILTER
# ============================================================

print("\nClassifying entrees...")

before_entree = len(df)

df = df[
    df.apply(is_entree, axis=1)
].copy()

entree_count = len(df)

print(f"Qualifying entrees: {entree_count:,}")
print(f"Removed non-entrees: {before_entree - entree_count:,}")

# ============================================================
# 7. REMOVE DUPLICATES
# ============================================================

print("\nRemoving duplicate recipes...")

before_duplicates = len(df)

# Remove duplicate recipe IDs.
df = df.drop_duplicates(
    subset=["RecipeId"],
    keep="first"
).copy()

# Remove recipes with the same name and ingredients.
df["_normalized_name"] = df["Name"].apply(normalize_text)

df["_ingredient_signature"] = (
    df["RecipeIngredientParts"].apply(ingredient_signature)
)

df = df.drop_duplicates(
    subset=[
        "_normalized_name",
        "_ingredient_signature"
    ],
    keep="first"
).copy()

# Remove temporary columns.
df = df.drop(
    columns=[
        "_normalized_name",
        "_ingredient_signature"
    ]
)

duplicate_count = len(df)

print(f"Recipes after deduplication: {duplicate_count:,}")
print(f"Duplicates removed: {before_duplicates - duplicate_count:,}")

# ============================================================
# 8. RANDOM SAMPLING
# ============================================================

print("\nSelecting random recipes...")

if len(df) >= TARGET_SIZE:

    final_df = df.sample(
        n=TARGET_SIZE,
        random_state=RANDOM_SEED
    ).copy()

else:

    print("\nWARNING:")
    print(
        f"Only {len(df):,} recipes passed all filters. "
        f"Saving all qualifying recipes instead."
    )

    final_df = df.copy()

# Shuffle the final dataset.
final_df = final_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

# ============================================================
# 9. SAVE OUTPUT
# ============================================================

print("\nSaving filtered dataset...")

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FILTERING COMPLETE")
print("=" * 60)

print(f"Original recipes:          {original_count:,}")
print(f"After quality filtering:   {quality_count:,}")
print(f"After entree filtering:    {entree_count:,}")
print(f"After deduplication:        {duplicate_count:,}")
print(f"Final recipes selected:    {len(final_df):,}")

print("\nFinal recipe categories:")

print(
    final_df["RecipeCategory"]
    .value_counts()
    .head(20)
    .to_string()
)

print(f"\nOutput file: {OUTPUT_FILE}")

print("=" * 60)
