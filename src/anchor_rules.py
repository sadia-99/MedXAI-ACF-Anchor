import pandas as pd


def compute_coverage(das, disease, binary_matrix, disease_col="Disease"):
    """
    Calcule la couverture d'un DAS pour une maladie.
    Couverture = nb patients ayant TOUS les symptômes du DAS
                 parmi les patients de la maladie
    """
    target = binary_matrix[
        binary_matrix[disease_col] == disease
    ].drop(columns=[disease_col])

    total = len(target)
    if total == 0:
        return 0.0

    # Vérifier que tous les symptômes du DAS sont dans la matrice
    valid_das = [s for s in das if s in target.columns]
    if not valid_das:
        return 0.0

    covered = (target[valid_das].sum(axis=1) == len(valid_das)).sum()
    return round(covered / total, 2)


def generate_rules(das_dict, binary_matrix, disease_col="Disease"):
    """
    Étape 5 — Génération des règles Anchor depuis les DAS.
    
    Pour chaque DAS → règle IF...THEN + couverture.
    Triées par couverture décroissante.
    """
    print("Étape 5 — Génération des règles Anchor...\n")
    rows = []

    for disease, das_list in das_dict.items():
        print(f"--- {disease} ---")

        for i, das in enumerate(das_list):
            coverage = compute_coverage(
                das, disease, binary_matrix, disease_col
            )
            conditions = " AND ".join([s.strip() for s in das])
            rule = f"IF {conditions} THEN {disease}"

            rows.append({
                "Maladie": disease,
                "Règle": rule,
                "DAS": das,
                "Couverture": coverage
            })

            print(f"  R{i+1}: {rule}")
            print(f"       Couverture = {coverage}")

    df_rules = pd.DataFrame(rows)
    df_rules = df_rules.sort_values(
        ["Maladie", "Couverture"],
        ascending=[True, False]
    ).reset_index(drop=True)

    print(f"\n✅ {len(df_rules)} règles générées au total")
    return df_rules


def save_rules(df_rules, path="results/rules/anchor_rules.csv"):
    """Sauvegarde les règles dans un fichier CSV."""
    df_rules.to_csv(path, index=False)
    print(f" Règles sauvegardées dans {path}")