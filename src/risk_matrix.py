"""
ISO 21448 SOTIF-inspired risk matrix for severity × likelihood scoring.

Severity  (S): 1–4  (1 = negligible, 4 = catastrophic)
Likelihood (L): 1–4  (1 = improbable, 4 = frequent)
Risk Score    : S × L  (1–16)

Risk bands:
  1–3   Acceptable
  4–8   ALARP / review
  9–16  Unacceptable
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List

import pandas as pd


class Severity(IntEnum):
    NEGLIGIBLE = 1    # No injury / minor damage
    MINOR = 2         # Minor injury / repairable damage
    SERIOUS = 3       # Serious injury / major damage
    CATASTROPHIC = 4  # Fatality / total loss


class Likelihood(IntEnum):
    IMPROBABLE = 1    # Rare edge case
    REMOTE = 2        # Occasional in adversarial scenarios
    OCCASIONAL = 3    # Appears in standard driving
    FREQUENT = 4      # Appears regularly in training


class RiskBand(str):
    ACCEPTABLE = "Acceptable"
    ALARP = "ALARP"
    UNACCEPTABLE = "Unacceptable"


def risk_band(score: int) -> str:
    if score <= 3:
        return RiskBand.ACCEPTABLE
    if score <= 8:
        return RiskBand.ALARP
    return RiskBand.UNACCEPTABLE


@dataclass
class RiskEntry:
    scenario: str
    failure_category: str
    severity: Severity
    likelihood: Likelihood

    @property
    def score(self) -> int:
        return int(self.severity) * int(self.likelihood)

    @property
    def band(self) -> str:
        return risk_band(self.score)


# Default severity mapping per failure category (ISO 21448 / SOTIF guidance)
DEFAULT_SEVERITY: Dict[str, Severity] = {
    "Perception": Severity.CATASTROPHIC,   # unseen obstacles → direct crash
    "Prediction": Severity.SERIOUS,        # wrong future state → near-miss / crash
    "Planning": Severity.SERIOUS,          # bad path → traffic violation / crash
    "Alignment": Severity.MINOR,           # proxy gaming → inefficiency / risky speed
    "Robustness": Severity.SERIOUS,        # OOD → unpredictable behaviour
    "Interaction": Severity.MINOR,         # deadlock → delay, occasionally escalates
    "Unknown": Severity.MINOR,
}


class RiskMatrix:
    def __init__(self) -> None:
        self._entries: List[RiskEntry] = []

    def add(
        self,
        scenario: str,
        failure_category: str,
        likelihood: Likelihood,
        severity: Severity | None = None,
    ) -> RiskEntry:
        sev = severity or DEFAULT_SEVERITY.get(failure_category, Severity.MINOR)
        entry = RiskEntry(scenario, failure_category, sev, likelihood)
        self._entries.append(entry)
        return entry

    def add_from_counts(
        self,
        scenario: str,
        category_counts: Dict[str, int],
        total_episodes: int,
    ) -> None:
        """Derive likelihood from failure rate and add one entry per category."""
        for category, count in category_counts.items():
            if count == 0:
                continue
            rate = count / max(total_episodes, 1)
            if rate < 0.05:
                lk = Likelihood.IMPROBABLE
            elif rate < 0.15:
                lk = Likelihood.REMOTE
            elif rate < 0.40:
                lk = Likelihood.OCCASIONAL
            else:
                lk = Likelihood.FREQUENT
            self.add(scenario, category, lk)

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "Scenario": e.scenario,
                "Failure Category": e.failure_category,
                "Severity": e.severity.name,
                "Severity (S)": int(e.severity),
                "Likelihood": e.likelihood.name,
                "Likelihood (L)": int(e.likelihood),
                "Risk Score": e.score,
                "Risk Band": e.band,
            }
            for e in self._entries
        ]
        return pd.DataFrame(rows)

    def summary(self) -> pd.DataFrame:
        df = self.to_dataframe()
        if df.empty:
            return df
        return (
            df.groupby(["Failure Category", "Risk Band"])
            .agg(Count=("Risk Score", "count"), Max_Score=("Risk Score", "max"))
            .reset_index()
            .sort_values("Max_Score", ascending=False)
        )

    def unacceptable_entries(self) -> List[RiskEntry]:
        return [e for e in self._entries if e.band == RiskBand.UNACCEPTABLE]
