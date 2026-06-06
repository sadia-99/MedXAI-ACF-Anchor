import pandas as pd
import numpy as np
from itertools import combinations


class FormalContext:
    """
    Analyse de Concepts Formels (ACF) pour le calcul des
    Ensembles Décisifs d'Attributs (DAS) — basé sur le PFE UMMTO 2022.
    """

    def __init__(self, df, disease_col="Disease"):
        """
        df          : DataFrame avec colonnes Symptom_1..Symptom_17 + Disease
        disease_col : nom de la colonne cible
        """
        self.df = df.copy()
        self.disease_col = disease_col
        self.binary_matrix = None
        self.symptoms = None
        self.diseases = None

    def build_context(self):
        """
        Transforme le dataset en matrice binaire (contexte formel).
        Ligne = patient, Colonne = symptôme, Valeur = 0 ou 1.
        """
        print("Construction du contexte formel...")

        # Récupérer toutes les colonnes symptômes
        symptom_cols = [c for c in self.df.columns if c.startswith("Symptom_")]

        # Extraire tous les symptômes uniques
        all_symptoms = set()
        for col in symptom_cols:
            all_symptoms.update(self.df[col].dropna().unique())
        self.symptoms = sorted(list(all_symptoms))

        # Construire la matrice binaire
        rows = []
        for _, row in self.df.iterrows():
            patient_symptoms = set(row[symptom_cols].dropna().values)
            binary_row = [1 if s in patient_symptoms else 0
                          for s in self.symptoms]
            rows.append(binary_row)

        self.binary_matrix = pd.DataFrame(
            rows,
            columns=self.symptoms
        )
        self.binary_matrix[self.disease_col] = self.df[self.disease_col].values
        self.diseases = self.df[self.disease_col].unique().tolist()

        print(f"Contexte formel construit :")
        print(f"   {len(self.binary_matrix)} patients")
        print(f"   {len(self.symptoms)} symptômes uniques")
        print(f"   {len(self.diseases)} maladies")

        return self.binary_matrix

    def compute_das(self, disease):
        """
        Calcule les Ensembles Décisifs d'Attributs (DAS) pour une maladie.
        Retourne les sous-ensembles minimaux de symptômes qui caractérisent
        uniquement cette maladie.
        """
        if self.binary_matrix is None:
            raise ValueError("Lance d'abord build_context()")

        print(f"\nCalcul des DAS pour : {disease}")

        # Patients de la maladie cible
        target = self.binary_matrix[
            self.binary_matrix[self.disease_col] == disease
        ].drop(columns=[self.disease_col])

        # Patients des autres maladies
        others = self.binary_matrix[
            self.binary_matrix[self.disease_col] != disease
        ].drop(columns=[self.disease_col])

        # Symptômes présents chez AU MOINS UN patient de la maladie
        relevant_symptoms = [s for s in self.symptoms
                             if target[s].sum() > 0]

        das_list = []

        # Chercher les sous-ensembles minimaux (du plus petit au plus grand)
        for size in range(1, len(relevant_symptoms) + 1):
            for combo in combinations(relevant_symptoms, size):
                combo = list(combo)

                # Vérifier que ce combo couvre au moins 1 patient cible
                covers_target = (target[combo].sum(axis=1) == len(combo)).any()
                if not covers_target:
                    continue

                # Vérifier que ce combo n'est PAS redondant avec un DAS déjà trouvé
                is_superset = any(
                    set(das).issubset(set(combo)) for das in das_list
                )
                if is_superset:
                    continue

                das_list.append(combo)

            # Arrêter si on a trouvé des DAS à cette taille
            if das_list and size >= 2:
                break

        print(f"{len(das_list)} DAS trouvés pour {disease}")
        return das_list