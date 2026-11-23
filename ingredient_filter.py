
import pandas as pd
import re
import ast
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

RECIPE_FILE = r"C:\Users\samjf\Downloads\ML LLM Project\ML_Project_AutoMeal\recipes_10000_entrees.csv"
WALMART_FILE = r"C:\Users\samjf\Downloads\ML LLM Project\ML_Project_AutoMeal\walmart_grocery.csv"

OUTPUT_FILE = "walmart_ingredients_filtered.csv"
UNMATCHED_FILE = "unmatched_ingredients.csv"
AUDIT_FILE = "walmart_ingredient_match_audit.csv"

# Set these manually if automatic column detection picks
# the wrong columns.
PRODUCT_NAME_COLUMN = None
PRICE_COLUMN = None

# Ingredient matching settings
MIN_INGREDIENT_LENGTH = 2

# Preparation words that should not affect matching.
PREPARATION_WORDS = {
    "fresh", "dried", "chopped", "diced", "minced",
    "sliced", "grated", "shredded", "peeled",
    "seeded", "crushed", "ground", "finely",
    "coarsely", "roughly", "thinly", "thickly",
    "large", "small", "medium", "boneless",
    "skinless", "cooked", "uncooked", "raw",
    "softened", "melted", "divided", "optional",
    "packed", "loosely", "firmly", "room",
    "temperature", "ripe", "chopped", "halved",
    "quartered", "sliced", "to", "taste",
    "as", "needed", "for", "garnish"
}

# Common words that don't help identify an ingredient.
STOP_WORDS = {
    "of", "and", "or", "the", "a", "an",
    "in", "with", "without", "plus"
}

# Optional ingredient synonyms.
# Add more as you inspect the matching results.
SYNONYMS = {
    "scallion": "green onion",
    "scallions": "green onion",
    "spring onion": "green onion",
    "garbanzo bean": "chickpea",
    "garbanzo beans": "chickpea",
    "courgette": "zucchini",
    "aubergine": "eggplant",
    "cilantro leaves": "cilantro",
    "confectioners sugar": "powdered sugar",
    "icing sugar": "powdered sugar",
}


# ============================================================
# 1. HELPER FUNCTIONS
# ============================================================

def parse_list(value):
    """Parse Python lists and Food.com R-style vectors."""

    if isinstance(value, list):
        return value

    if not isinstance(value, str):
        return []

    value = value.strip()

    if not value:
        return []

    if value.startswith("c(") and value.endswith(")"):
        inner = value[2:-1].strip()

        try:
            result = ast.literal_eval("[" + inner + "]")
            return result if isinstance(result, list) else []
        except (ValueError, SyntaxError):
            return []

    try:
        result = ast.literal_eval(value)

        if isinstance(result, list):
            return result

    except (ValueError, SyntaxError):
        pass

    return []


def singularize(word):
    """Basic English plural normalization."""

    if len(word) <= 3:
        return word

    irregular = {
        "tomatoes": "tomato",
        "potatoes": "potato",
        "leaves": "leaf",
        "berries": "berry",
        "cherries": "cherry",
        "loaves": "loaf",
        "knives": "knife",
        "halves": "half",
        "teeth": "tooth",
        "feet": "foot",
        "children": "child",
    }

    if word in irregular:
        return irregular[word]

    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"

    if word.endswith("sses"):
        return word[:-2]

    if word.endswith("s") and not word.endswith(
        ("ss", "us", "is")
    ):
        return word[:-1]

    return word


def normalize_ingredient(value):
    """
    Normalize recipe ingredients and Walmart product names.

    Removes preparation descriptors and normalizes
    singular/plural forms.
    """

    if value is None:
        return ""

    text = str(value).lower().strip()

    # Remove HTML-like entities and punctuation.
    text = re.sub(r"&amp;", " and ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove numeric quantities.
    text = re.sub(r"\b\d+\b", " ", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    # Apply known synonyms before token processing.
    if text in SYNONYMS:
        text = SYNONYMS[text]

    words = text.split()

    cleaned = []

    for word in words:

        if word in PREPARATION_WORDS:
            continue

        if word in STOP_WORDS:
            continue

        word = singularize(word)

        if word:
            cleaned.append(word)

    result = " ".join(cleaned)

    return result


def parse_price(value):
    """
    Convert prices into numeric values.

    Examples:
        $3.98 -> 3.98
        4.50 -> 4.50
        N/A -> None
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    # Find the first numeric price.
    match = re.search(r"\d+(?:,\d{3})*(?:\.\d+)?", value)

    if not match:
        return None

    try:
        price = float(match.group().replace(",", ""))

        if price <= 0:
            return None

        return price

    except ValueError:
        return None


def find_column(df, candidates, description):
    """Find a matching column regardless of capitalization."""

    lookup = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise ValueError(
        f"Could not find {description} column.\n"
        f"Available columns: {df.columns.tolist()}\n"
        f"Please set the appropriate column manually."
    )


# ============================================================
# 2. LOAD RECIPE DATA
# ============================================================

print("=" * 65)
print("WALMART INGREDIENT FILTERING")
print("=" * 65)

print("\nLoading recipe dataset...")

recipes = pd.read_csv(
    RECIPE_FILE,
    low_memory=False
)

recipes.columns = (
    recipes.columns
    .str.strip()
    .str.replace("\ufeff", "", regex=False)
)

if "RecipeIngredientParts" not in recipes.columns:
    raise ValueError(
        "RecipeIngredientParts column not found."
    )

print(f"Loaded recipes: {len(recipes):,}")

# Parse ingredients.
recipes["RecipeIngredientParts"] = (
    recipes["RecipeIngredientParts"].apply(parse_list)
)


# ============================================================
# 3. EXTRACT UNIQUE RECIPE INGREDIENTS
# ============================================================

print("\nExtracting recipe ingredients...")

ingredient_counts = defaultdict(int)
ingredient_originals = defaultdict(set)

for ingredients in recipes["RecipeIngredientParts"]:

    for ingredient in ingredients:

        if not isinstance(ingredient, str):
            continue

        ingredient = ingredient.strip()

        if not ingredient:
            continue

        normalized = normalize_ingredient(ingredient)

        if len(normalized) < MIN_INGREDIENT_LENGTH:
            continue

        ingredient_counts[normalized] += 1
        ingredient_originals[normalized].add(ingredient)


# Build ingredient list.
recipe_ingredients = sorted(
    ingredient_counts.keys()
)

print(f"Unique normalized ingredients: {len(recipe_ingredients):,}")

# Save ingredient reference list.
ingredient_reference = pd.DataFrame([
    {
        "NormalizedIngredient": ingredient,
        "OriginalIngredientExamples": " | ".join(
            sorted(ingredient_originals[ingredient])[:5]
        ),
        "RecipeFrequency": ingredient_counts[ingredient],
    }
    for ingredient in recipe_ingredients
])

ingredient_reference.to_csv(
    "recipe_ingredient_reference.csv",
    index=False
)


# ============================================================
# 4. LOAD WALMART DATA
# ============================================================

print("\nLoading Walmart products...")

walmart = pd.read_csv(
    WALMART_FILE,
    low_memory=False
)

walmart.columns = (
    walmart.columns
    .str.strip()
    .str.replace("\ufeff", "", regex=False)
)

print(f"Original Walmart products: {len(walmart):,}")

# Automatically detect product name column.
if PRODUCT_NAME_COLUMN is None:

    PRODUCT_NAME_COLUMN = find_column(
        walmart,
        [
            "Product Name",
            "Product title",
            "Product",
            "product_name",
            "SKU_Name",
            "name",
            "title",
            "Item Name",
        ],
        "product name"
    )

# Automatically detect price column.
if PRICE_COLUMN is None:

    PRICE_COLUMN = find_column(
        walmart,
        [
            "Sale Price",
            "Online price",
            "Price",
            "product_price",
            "final_price",
            "price",
            "List Price",
            "PRICE_RETAIL",
        ],
        "price"
    )

print(f"Product name column: {PRODUCT_NAME_COLUMN}")
print(f"Price column: {PRICE_COLUMN}")


# ============================================================
# 5. CLEAN WALMART PRODUCT DATA
# ============================================================

print("\nCleaning Walmart products...")

walmart = walmart.dropna(
    subset=[PRODUCT_NAME_COLUMN]
).copy()

walmart["_normalized_product"] = (
    walmart[PRODUCT_NAME_COLUMN].apply(normalize_ingredient)
)

walmart["_numeric_price"] = (
    walmart[PRICE_COLUMN].apply(parse_price)
)

# Remove products without valid names or prices.
walmart = walmart[
    walmart["_normalized_product"].ne("")
    & walmart["_numeric_price"].notna()
].copy()

print(
    f"Products with valid names and prices: {len(walmart):,}"
)


# ============================================================
# 6. BUILD PRODUCT SEARCH INDEX
# ============================================================

print("\nBuilding product matching index...")

product_tokens = {}

# Maps each ingredient token to product row indices.
token_index = defaultdict(set)

for idx, normalized in walmart["_normalized_product"].items():

    tokens = set(normalized.split())

    product_tokens[idx] = tokens

    for token in tokens:
        token_index[token].add(idx)

print("Product index created.")


# ============================================================
# 7. MATCH WALMART PRODUCTS TO RECIPE INGREDIENTS
# ============================================================

print("\nMatching Walmart products to recipe ingredients...")

matches = []
unmatched = []

# Match longer/more specific ingredients first.
sorted_ingredients = sorted(
    recipe_ingredients,
    key=lambda x: len(x.split()),
    reverse=True
)

for ingredient in sorted_ingredients:

    tokens = set(ingredient.split())

    if not tokens:
        continue

    # Find products containing every ingredient token.
    candidate_sets = [
        token_index.get(token, set())
        for token in tokens
    ]

    if any(not candidates for candidates in candidate_sets):
        unmatched.append(ingredient)
        continue

    candidate_indices = set.intersection(*candidate_sets)

    if not candidate_indices:
        unmatched.append(ingredient)
        continue

    # Retain every potential match for the audit file.
    for idx in candidate_indices:

        product = walmart.loc[idx]

        matches.append({
            "MatchedIngredient": ingredient,
            "OriginalIngredientExamples": " | ".join(
                sorted(ingredient_originals[ingredient])[:5]
            ),
            "RecipeFrequency": ingredient_counts[ingredient],
            "ProductName": product[PRODUCT_NAME_COLUMN],
            "ProductPrice": product["_numeric_price"],
            "WalmartRowIndex": idx,
        })


# ============================================================
# 8. SAVE MATCHING AUDIT
# ============================================================

print("\nSaving matching audit...")

if matches:

    audit_df = pd.DataFrame(matches)

    audit_df.to_csv(
        AUDIT_FILE,
        index=False
    )

else:

    audit_df = pd.DataFrame()

    print("WARNING: No products matched.")


# ============================================================
# 9. SELECT THE CHEAPEST PRODUCT PER INGREDIENT
# ============================================================

print("\nSelecting cheapest product for each ingredient...")

if not audit_df.empty:

    # Sort by ingredient and price.
    audit_df = audit_df.sort_values(
        by=["MatchedIngredient", "ProductPrice"],
        ascending=[True, True]
    )

    # Select the cheapest Walmart product for each ingredient.
    cheapest = audit_df.drop_duplicates(
        subset=["MatchedIngredient"],
        keep="first"
    ).copy()

    # Reset indices to avoid alignment issues.
    cheapest = cheapest.reset_index(drop=True)

    # Retrieve original Walmart product rows.
    final_df = walmart.loc[
        cheapest["WalmartRowIndex"].values
    ].copy()

    # Reset the product DataFrame index.
    final_df = final_df.reset_index(drop=True)

    # Assign matching information row by row.
    final_df["MatchedIngredient"] = (
        cheapest["MatchedIngredient"].values
    )

    final_df["OriginalIngredientExamples"] = (
        cheapest["OriginalIngredientExamples"].values
    )

    final_df["RecipeFrequency"] = (
        cheapest["RecipeFrequency"].values
    )

    final_df["SelectedPrice"] = (
        final_df["_numeric_price"]
    )

    # Remove internal helper columns.
    final_df = final_df.drop(
        columns=[
            "_normalized_product",
            "_numeric_price",
        ]
    )

    # Save final dataset.
    final_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"Successfully selected {len(final_df):,} "
        f"ingredient-product matches."
    )

else:

    final_df = pd.DataFrame()

    print("No matching products were found.")


# ============================================================
# 10. SAVE UNMATCHED INGREDIENTS
# ============================================================

print("\nSaving unmatched ingredients...")

unmatched_df = ingredient_reference[
    ingredient_reference["NormalizedIngredient"].isin(unmatched)
].copy()

unmatched_df.to_csv(
    UNMATCHED_FILE,
    index=False
)


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("WALMART INGREDIENT FILTERING COMPLETE")
print("=" * 65)

print(f"Original recipes:              {len(recipes):,}")
print(f"Unique recipe ingredients:     {len(recipe_ingredients):,}")
print(f"Original Walmart products:     {len(pd.read_csv(WALMART_FILE, usecols=[PRODUCT_NAME_COLUMN])):,}")
print(f"Valid Walmart products:        {len(walmart):,}")
print(f"Ingredient matches:            {len(matches):,}")

print(
    f"Unique ingredients matched:    "
    f"{len(final_df):,}"
)

print(
    f"Unmatched ingredients:         "
    f"{len(unmatched_df):,}"
)

print("\nOUTPUT FILES:")

print(f"1. {OUTPUT_FILE}")
print(f"2. {UNMATCHED_FILE}")
print(f"3. {AUDIT_FILE}")
print("4. recipe_ingredient_reference.csv")

print("=" * 65)
