
"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    df = pd.read_csv(data.path)

    missing = df.isna().sum().to_dict()

    over_30 = int((df['Age'] > 30).sum())

    mean_age = df.groupby('Pclass')['Age'].mean()
    mean_age = {int(key): value for key, value in mean_age.items()}

    survival_rate = df.groupby('Pclass')['Survived'].mean()
    survival_rate = {
        int(key): value for key, value in survival_rate.items()
    }

    top_fares = df['Fare'].nlargest(5).tolist()

    return TitanicSummary(
        missing_values=missing,
        passengers_over_30=over_30,
        mean_age_by_class=mean_age,
        survival_rate_by_class=survival_rate,
        top_fares=top_fares
    )
