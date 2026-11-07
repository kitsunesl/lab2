"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    df = pd.read_csv(data.path, sep='\t', na_values='NA')

    features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']

    men = df[df['Gender'] == 'Male']
    women = df[df['Gender'] == 'Female']

    male_correlations = men[features].corrwith(men['MRI_Count']).to_dict()
    female_correlations = women[features].corrwith(women['MRI_Count']).to_dict()

    strongest = max(
        features,
        key=lambda feature: max(
            abs(male_correlations[feature]),
            abs(female_correlations[feature])
        )
    )

    return BrainCorrelationSummary(
        male_correlations=male_correlations,
        female_correlations=female_correlations,
        strongest_mri_feature=strongest
    )
