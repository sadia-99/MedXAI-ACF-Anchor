import pandas as pd
import numpy as np


class FormalContext:

    def __init__(self, df, disease_col="Disease"):
        self.df = df.copy()
        self.disease_col = disease_col
        self.KOMR = None
        self.symptoms = None
        self.diseases = None
        self.concepts = []

    def build_context(self):
        print("Étape 1 — Transformation multi-label → contexte formel...")
        symptom_cols = [c for c in self.df.columns if c.startswith("Symptom_")]
        all_symptoms = set()
        for col in symptom_cols:
            all_symptoms.update(self.df[col].dropna().unique())
        self.symptoms = sorted([s.strip() for s in all_symptoms])
        self.diseases = self.df[self.disease_col].unique().tolist()

        patient_symptom = pd.DataFrame(0,
            index=range(len(self.df)),
            columns=self.symptoms)
        for idx, row in self.df.iterrows():
            patient_syms = set(s.strip() for s in row[symptom_cols].dropna().values)
            for s in patient_syms:
                if s in self.symptoms:
                    patient_symptom.loc[idx, s] = 1

        patient_disease = pd.DataFrame(0,
            index=range(len(self.df)),
            columns=self.diseases)
        for idx, row in self.df.iterrows():
            patient_disease.loc[idx, row[self.disease_col]] = 1

        komr_cols = []
        for disease in self.diseases:
            for symptom in self.symptoms:
                komr_cols.append(f"{disease}_{symptom}")

        self.KOMR = pd.DataFrame(0,
            index=range(len(self.df)),
            columns=komr_cols)

        for idx in range(len(self.df)):
            for disease in self.diseases:
                if patient_disease.loc[idx, disease] == 1:
                    for symptom in self.symptoms:
                        if patient_symptom.loc[idx, symptom] == 1:
                            self.KOMR.loc[idx, f"{disease}_{symptom}"] = 1

        print(f"Contexte formel KOMR construit :")
        print(f"   {len(self.KOMR)} patients")
        print(f"   {len(self.diseases)} maladies")
        print(f"   {len(self.symptoms)} symptômes")
        print(f"   {len(komr_cols)} colonnes (Maladie_Symptôme)")
        return self.KOMR

    def extract_concepts(self):
        if self.KOMR is None:
            raise ValueError("Lance d'abord build_context()")

        print("\nÉtape 2 — Extraction concepts formels (algorithme Chein)...")

        L = []
        for idx in range(len(self.KOMR)):
            row = self.KOMR.iloc[idx]
            intension = frozenset(row[row == 1].index.tolist())
            if intension:
                L.append((frozenset([idx]), intension))

        all_concepts = []
        p = 1

        while len(L) > 1:
            print(f"   Niveau L{p} : {len(L)} rectangles")
            L_next = []
            marked_in_L = set()

            for i in range(len(L)):
                for j in range(i + 1, len(L)):
                    Xi, Yi = L[i]
                    Xj, Yj = L[j]
                    Yij = Yi & Yj

                    if not Yij:
                        continue

                    Xij = Xi | Xj
                    found = False
                    for k, (Xk, Yk) in enumerate(L_next):
                        if Yk == Yij:
                            L_next[k] = (Xk | Xij, Yk)
                            found = True
                            break

                    if not found:
                        L_next.append((Xij, Yij))

                    if Yij == Yi:
                        marked_in_L.add(i)
                    if Yij == Yj:
                        marked_in_L.add(j)

            for i, concept in enumerate(L):
                if i not in marked_in_L:
                    all_concepts.append(concept)

            L = L_next
            p += 1

        all_concepts.extend(L)
        self.concepts = all_concepts
        print(f"{len(self.concepts)} concepts formels extraits")
        return self.concepts