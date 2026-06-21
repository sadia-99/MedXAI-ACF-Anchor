# MedXAI — Explicabilité des modèles de classification multi-label
## Approche par Analyse de Concepts Formels (ACF) + Règles Anchor

**Contexte :** Les modèles de machine learning utilisés en médecine sont souvent des "boîtes noires". 
Ce notebook implémente une approche originale d'explicabilité basée sur :
1. L'**Analyse de Concepts Formels (ACF)** pour extraire les attributs décisifs (DAS)
2. Les **Règles Anchor** pour générer des explications lisibles par les médecins

**Dataset :** Symptômes cliniques → Maladies respiratoires (Bronchial Asthma, Tuberculosis, Pneumonia, Common Cold)

**Basé sur :** Mémoire de Master IA — Université UMMTO 2022