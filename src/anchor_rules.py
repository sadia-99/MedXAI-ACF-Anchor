import pandas as pd


def compute_coverage(das, disease, binary_matrix, disease_col="Disease"):
    """
    Calcule la couverture d'un DAS pour une maladie donnée.
    Couverture = nb patients couverts par le DAS / total patients de la maladie
    """
    # Patients de la maladie cible
    target = binary_matrix[
        binary_matrix[disease_col] == disease
    ].drop(columns=[disease_col])

    total = len(target)
    if total == 0:
        return 0.0

    # Patients couverts par le DAS (ont TOUS les symptômes du DAS)
    covered = (target[das].sum(axis=1) == len(das)).sum()

    return round(covered / total, 2)


def generate_rules(das_dict, binary_matrix, disease_col="Disease"):
    """
    Génère les règles Anchor à partir des DAS.
    
    das_dict : dictionnaire {maladie: [das1, das2, ...]}
    Retourne un DataFrame avec colonnes : Maladie, Règle, Couverture
    """
    print("Génération des règles Anchor...\n")
    rows = []

    for disease, das_list in das_dict.items():
        print(f"--- {disease} ---")

        for i, das in enumerate(das_list):
            # Calculer la couverture
            coverage = compute_coverage(
                das, disease, binary_matrix, disease_col
            )

            # Construire la règle lisible
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

    # Créer le DataFrame et trier par maladie + couverture décroissante
    df_rules = pd.DataFrame(rows)
    df_rules = df_rules.sort_values(
        ["Maladie", "Couverture"],
        ascending=[True, False]
    ).reset_index(drop=True)

    print(f"\n {len(df_rules)} règles générées au total")
    return df_rules


def save_rules(df_rules, path="results/rules/anchor_rules.csv"):
    """
    Sauvegarde les règles dans un fichier CSV.
    """
    df_rules.to_csv(path, index=False)
    print(f" Règles sauvegardées dans {path}")