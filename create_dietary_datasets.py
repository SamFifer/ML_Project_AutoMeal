
import pandas as pd
import ast
import re

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "Recipes/recipes_10000_entrees.csv"

OUTPUT_FILES = {
    "vegetarian": "Recipes/recipes_vegetarian.csv",
    "vegan": "Recipes/recipes_vegan.csv",
    "pescatarian": "Recipes/recipes_pescatarian.csv",
    "gluten_free": "Recipes/recipes_gluten_free.csv",
    "kosher": "Recipes/recipes_kosher.csv",
    "halal": "Recipes/recipes_halal.csv",
}

# ============================================================
# 1. INGREDIENT EXCLUSION LISTS
# ============================================================

# Ingredients that disqualify a vegetarian recipe.
MEAT_AND_SEAFOOD = [
    "beef",
    "pork",
    "ham",
    "bacon",
    "sausage",
    "pepperoni",
    "salami",
    "prosciutto",
    "chicken",
    "turkey",
    "duck",
    "goose",
    "quail",
    "pheasant",
    "lamb",
    "veal",
    "venison",
    "bison",
    "rabbit",
    "goat meat",
    "ground meat",
    "ground beef",
    "ground turkey",
    "ground pork",
    "meat",
    "fish",
    "salmon",
    "tuna",
    "cod",
    "tilapia",
    "anchovy",
    "anchovies",
    "sardine",
    "sardines",
    "shrimp",
    "prawn",
    "crab",
    "lobster",
    "clam",
    "clams",
    "mussel",
    "mussels",
    "oyster",
    "oysters",
    "scallop",
    "scallops",
    "squid",
    "octopus",
    "seafood",
    "fish sauce",
    "oyster sauce",
    "chicken broth",
    "chicken stock",
    "beef broth",
    "beef stock",
    "bone broth",
    "meat broth",
    "gelatin",
    "lard",
    "tallow",
    "suet",
    "kielbasa",
    "bratwurst",
    "chorizo",
    "hot dog",
]

# Additional animal-derived ingredients that disqualify
# a recipe from being vegan.
ANIMAL_PRODUCTS = [
    "milk",
    "whole milk",
    "skim milk",
    "cream",
    "heavy cream",
    "half and half",
    "butter",
    "ghee",
    "cheese",
    "parmesan",
    "mozzarella",
    "cheddar",
    "ricotta",
    "yogurt",
    "yoghurt",
    "sour cream",
    "buttermilk",
    "whey",
    "casein",
    "egg",
    "eggs",
    "egg white",
    "egg yolk",
    "mayonnaise",
    "honey",
    "beeswax",
    "gelatin",
    "lard",
    "tallow",
    "anchovy paste",
    "fish sauce",
    "oyster sauce",
]

# Ingredients that may contain gluten.
GLUTEN_INGREDIENTS = [
    "wheat",
    "wheat flour",
    "bread flour",
    "all purpose flour",
    "whole wheat",
    "barley",
    "rye",
    "malt",
    "malt vinegar",
    "malt extract",
    "bulgur",
    "couscous",
    "semolina",
    "farina",
    "durum",
    "spelt",
    "farro",
    "seitan",
    "vital wheat gluten",
    "breadcrumbs",
    "bread crumbs",
    "regular soy sauce",
    "teriyaki sauce",
    "beer",
    "ale",
]

# Meat and poultry exclusions for pescatarian recipes.
# Seafood is intentionally allowed.
PESCATARIAN_EXCLUSIONS = [
    "beef",
    "pork",
    "ham",
    "bacon",
    "sausage",
    "pepperoni",
    "salami",
    "prosciutto",
    "chicken",
    "turkey",
    "duck",
    "goose",
    "quail",
    "pheasant",
    "lamb",
    "veal",
    "venison",
    "bison",
    "rabbit",
    "goat meat",
    "ground meat",
    "ground beef",
    "ground turkey",
    "ground pork",
    "meat",
    "lard",
    "tallow",
    "suet",
    "gelatin",
    "chicken broth",
    "chicken stock",
    "beef broth",
    "beef stock",
    "bone broth",
    "meat broth",
    "kielbasa",
    "bratwurst",
    "chorizo",
    "hot dog",
]

# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def parse_list(value):
    """
    Parse Food.com ingredient and keyword columns.

    Supports:
        c("item 1", "item 2")
        ["item 1", "item 2"]
    """

    if isinstance(value, list):
        return value

    if not isinstance(value, str):
        return []

    value = value.strip()

    if not value:
        return []

    # Handle R-style vectors.
    if value.startswith("c(") and value.endswith(")"):
        inner = value[2:-1].strip()

        try:
            result = ast.literal_eval("[" + inner + "]")
            return result if isinstance(result, list) else []
        except (ValueError, SyntaxError):
            return []

    # Handle Python-style lists.
    try:
        result = ast.literal_eval(value)

        if isinstance(result, list):
            return result

    except (ValueError, SyntaxError):
        pass

    return []


def normalize_text(value):
    """Convert text to lowercase and normalize spacing."""

    if value is None:
        return ""

    value = str(value).lower().strip()
    value = re.sub(r"\s+", " ", value)

    return value


def contains_term(text, terms):
    """
    Search for complete words or phrases.
    Avoids matching fragments of unrelated words.
    """

    text = normalize_text(text)

    for term in terms:
        term = normalize_text(term)

        pattern = r"(?<!\w)" + re.escape(term) + r"s?(?!\w)"

        if re.search(pattern, text):
            return True

    return False


def get_ingredients(row):
    """Convert ingredient list into searchable text."""

    ingredients = row["RecipeIngredientParts"]

    if not isinstance(ingredients, list):
        return normalize_text(row["Name"])

    return normalize_text(row["Name"]) + " " + " ".join(
        normalize_text(item)
        for item in ingredients
        if item is not None
    )


def get_keywords(row):
    """Convert keywords into searchable text."""

    keywords = row["Keywords"]

    if not isinstance(keywords, list):
        return ""

    return " ".join(
        normalize_text(item)
        for item in keywords
        if item is not None
    )


# ============================================================
# 3. DIETARY CLASSIFICATION FUNCTIONS
# ============================================================

def is_vegetarian(row):
    """
    Vegetarian:
    No meat, poultry, fish, or seafood.
    """

    ingredients = get_ingredients(row)

    return not contains_term(
        ingredients,
        MEAT_AND_SEAFOOD
    )


def is_vegan(row):
    """
    Vegan:
    No animal-derived ingredients.
    """

    ingredients = get_ingredients(row)

    forbidden = MEAT_AND_SEAFOOD + ANIMAL_PRODUCTS

    return not contains_term(
        ingredients,
        forbidden
    )


def is_pescatarian(row):
    """
    Pescatarian:
    No meat or poultry.
    Fish and seafood are allowed.
    """

    ingredients = get_ingredients(row)

    return not contains_term(
        ingredients,
        PESCATARIAN_EXCLUSIONS
    )


def is_gluten_free(row):
    """
    Gluten-free:
    Requires an explicit gluten-free keyword.
    Also checks ingredients for obvious gluten sources.
    """

    keywords = get_keywords(row)
    ingredients = get_ingredients(row)

    gluten_free_labels = [
        "gluten free",
        "gluten-free",
        "glutenfree",
        "celiac friendly",
        "coeliac friendly",
    ]

    # Must have an explicit gluten-free designation.
    if not contains_term(keywords, gluten_free_labels):
        return False

    # Reject obvious gluten-containing ingredients.
    if contains_term(ingredients, GLUTEN_INGREDIENTS):
        return False

    return True


def is_kosher(row):
    """
    Kosher:
    Requires an explicit kosher designation in keywords.
    """

    keywords = get_keywords(row)

    kosher_labels = [
        "kosher",
        "kosher style",
    ]

    return contains_term(keywords, kosher_labels)


def is_halal(row):
    """
    Halal:
    Requires an explicit halal designation in keywords.
    """

    keywords = get_keywords(row)

    halal_labels = [
        "halal",
        "halal friendly",
    ]

    return contains_term(keywords, halal_labels)


# ============================================================
# 4. LOAD THE 10,000 RECIPE DATASET
# ============================================================

print("=" * 60)
print("CREATING DIETARY RECIPE DATASETS")
print("=" * 60)

print("\nLoading original filtered recipes...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

df.columns = (
    df.columns
    .str.strip()
    .str.replace("\ufeff", "", regex=False)
)

print(f"Loaded recipes: {len(df):,}")

# Make sure the necessary columns exist.
required_columns = [
    "Name",
    "RecipeIngredientParts",
    "Keywords",
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

# Parse the columns used for dietary classification.
print("\nParsing ingredients and keywords...")

df["RecipeIngredientParts"] = (
    df["RecipeIngredientParts"].apply(parse_list)
)

df["Keywords"] = df["Keywords"].apply(parse_list)

print("Parsing complete.")


# ============================================================
# 5. CREATE THE SIX INDEPENDENT DATASETS
# ============================================================

dietary_filters = {
    "vegetarian": is_vegetarian,
    "vegan": is_vegan,
    "pescatarian": is_pescatarian,
    "gluten_free": is_gluten_free,
    "kosher": is_kosher,
    "halal": is_halal,
}

results = {}

for diet, filter_function in dietary_filters.items():

    print(f"\nCreating {diet} dataset...")

    # Always filter from the original 10,000 recipes.
    subset = df[
        df.apply(filter_function, axis=1)
    ].copy()

    # Save the independent dataset.
    output_file = OUTPUT_FILES[diet]

    subset.to_csv(
        output_file,
        index=False
    )

    results[diet] = len(subset)

    print(f"Recipes found: {len(subset):,}")
    print(f"Saved to: {output_file}")


# ============================================================
# 6. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DIETARY DATASETS COMPLETE")
print("=" * 60)

print(f"Original dataset: {len(df):,} recipes\n")

for diet, count in results.items():

    percentage = (
        count / len(df) * 100
        if len(df) > 0
        else 0
    )

    print(
        f"{diet.capitalize():15} "
        f"{count:>7,} recipes "
        f"({percentage:.1f}%)"
    )

print("\nSix independent files created successfully.")

print("=" * 60)
