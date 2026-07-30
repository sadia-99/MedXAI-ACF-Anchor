import pandas as pd
import numpy as np


class FormalContext:
    """
    Étape 1 — Transformation des données multi-label en contexte formel.
    
    Données 3D (Patient × Symptôme × Maladie)
    → Contexte formel 2D KOMR (Patient × (Maladie-Symptôme))
    
    Règle : KOMR[patient][maladie_symptôme] = 1
            si patient a le symptôme ET est diagnostiqué avec la maladie
    """

    def __init__(self, df, disease_col="Disease"):
        self.df = df.copy()
        self.disease_col = disease_col
        self.KOMR = None
        self.symptoms = None
        self.diseases = None

    def build_context(self):
        """
        Construit la matrice KOMR (Patient × Maladie-Symptôme).
        """
        print("Étape 1 — Transformation multi-label → contexte formel...")

        # Récupérer les colonnes symptômes
        symptom_cols = [c for c in self.df.columns if c.startswith("Symptom_")]

        # Extraire tous les symptômes uniques
        all_symptoms = set()
        for col in symptom_cols:
            all_symptoms.update(self.df[col].dropna().unique())
        self.symptoms = sorted([s.strip() for s in all_symptoms])
        self.diseases = self.df[self.disease_col].unique().tolist()

        # Matrice intermédiaire Patient × Symptôme
        patient_symptom = pd.DataFrame(0,
            index=range(len(self.df)),
            columns=self.symptoms)

        for idx, row in self.df.iterrows():
            patient_syms = set(
                s.strip() for s in row[symptom_cols].dropna().values
            )
            for s in patient_syms:
                if s in self.symptoms:
                    patient_symptom.loc[idx, s] = 1

        # Matrice intermédiaire Patient × Maladie
        patient_disease = pd.DataFrame(0,
            index=range(len(self.df)),
            columns=self.diseases)

        for idx, row in self.df.iterrows():
            patient_disease.loc[idx, row[self.disease_col]] = 1

        # Construction de KOMR — Patient × (Maladie_Symptôme)
        komr_cols = []
        for disease in self.diseases:
            for symptom in self.symptoms:
                komr_cols.append(f"{disease}_{symptom}")

        self.KOMR = pd.DataFrame(0,
            index=range(len(self.df)),
            columns=komr_cols)

        # Remplissage : 1 si patient a symptôme ET maladie
        for idx in range(len(self.df)):
            for disease in self.diseases:
                if patient_disease.loc[idx, disease] == 1:
                    for symptom in self.symptoms:
                        if patient_symptom.loc[idx, symptom] == 1:
                            col = f"{disease}_{symptom}"
                            self.KOMR.loc[idx, col] = 1

        print(f"Contexte formel KOMR construit :")
        print(f"   {len(self.KOMR)} patients (lignes)")
        print(f"   {len(self.diseases)} maladies")
        print(f"   {len(self.symptoms)} symptômes")
        print(f"   {len(komr_cols)} colonnes (Maladie_Symptôme)")
        print(f"\nAperçu KOMR (5 premières colonnes) :")
        print(self.KOMR.iloc[:5, :5])

        return self.KOMR