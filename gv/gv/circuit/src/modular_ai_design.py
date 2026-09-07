#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块化元件设计与AI表达预测。

该模块用于：
1. 读取并管理启动子/RBS/CDS/终止子元件数据；
2. 将元件组合映射为数值特征；
3. 使用轻量级岭回归模型预测组合表达表现；
4. 将组合参数映射到GasVesicleCircuit并执行动力学仿真。
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModularElement:
    """???????"""

    element_id: str
    element_type: str
    species: str
    strength: float
    sequence: str
    description: str


class ModularElementLibrary:
    """????????????"""

    FILE_MAP = {
        "promoter": "promoters.csv",
        "rbs": "rbs.csv",
        "cds": "cds.csv",
        "terminator": "terminators.csv",
    }

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.tables = self._load_tables()

    def _load_tables(self) -> Dict[str, pd.DataFrame]:
        tables: Dict[str, pd.DataFrame] = {}
        for element_type, filename in self.FILE_MAP.items():
            path = os.path.join(self.data_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"?????????: {path}")

            table = pd.read_csv(path)
            required_cols = {"element_id", "species", "strength", "sequence", "description"}
            if not required_cols.issubset(set(table.columns)):
                raise ValueError(f"{filename} ??????: {required_cols}")

            table = table.copy()
            table["element_type"] = element_type
            tables[element_type] = table
        return tables

    def element_options(self, element_type: str) -> List[str]:
        return self.tables[element_type]["element_id"].tolist()

    def get_element(self, element_type: str, element_id: str) -> ModularElement:
        table = self.tables[element_type]
        row = table.loc[table["element_id"] == element_id]
        if row.empty:
            raise ValueError(f"?????: {element_type}/{element_id}")
        data = row.iloc[0]
        return ModularElement(
            element_id=data["element_id"],
            element_type=element_type,
            species=data["species"],
            strength=float(data["strength"]),
            sequence=str(data["sequence"]),
            description=str(data["description"]),
        )

    def build_design_features(self, design: Dict[str, str]) -> np.ndarray:
        promoter = self.get_element("promoter", design["promoter"])
        rbs = self.get_element("rbs", design["rbs"])
        cds = self.get_element("cds", design["cds"])
        terminator = self.get_element("terminator", design["terminator"])

        strengths = np.array([
            promoter.strength,
            rbs.strength,
            cds.strength,
            terminator.strength,
        ])
        unique_species = len({promoter.species, rbs.species, cds.species, terminator.species})
        species_penalty = (unique_species - 1) * 0.12

        # ??????????????????
        feature_vec = np.array(
            [
                strengths[0],
                strengths[1],
                strengths[2],
                strengths[3],
                strengths[0] * strengths[1],
                strengths[2] * strengths[3],
                strengths[0] * strengths[2],
                species_penalty,
                1.0,  # bias
            ],
            dtype=float,
        )
        return feature_vec

    def generate_training_set(
        self,
        n_samples: int = 240,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """??????????????????"""
        rng = np.random.default_rng(random_state)

        promoter_ids = self.element_options("promoter")
        rbs_ids = self.element_options("rbs")
        cds_ids = self.element_options("cds")
        terminator_ids = self.element_options("terminator")

        records = []
        for _ in range(n_samples):
            design = {
                "promoter": promoter_ids[rng.integers(0, len(promoter_ids))],
                "rbs": rbs_ids[rng.integers(0, len(rbs_ids))],
                "cds": cds_ids[rng.integers(0, len(cds_ids))],
                "terminator": terminator_ids[rng.integers(0, len(terminator_ids))],
            }
            x = self.build_design_features(design)
            noise = rng.normal(0.0, 0.05)
            y = (
                0.55 * x[0]
                + 0.70 * x[1]
                + 0.52 * x[2]
                + 0.40 * x[3]
                + 0.34 * x[4]
                + 0.26 * x[5]
                - 0.48 * x[7]
                + noise
            )
            records.append(
                {
                    **design,
                    "predicted_target": float(max(0.02, y)),
                }
            )

        return pd.DataFrame(records)


class RidgeExpressionPredictor:
    """???????????????ML????"""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.coefficients: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        xtx = X.T @ X
        reg = self.alpha * np.eye(xtx.shape[0])
        self.coefficients = np.linalg.solve(xtx + reg, X.T @ y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise ValueError("模型尚未训练")
        return X @ self.coefficients

    def score_r2(self, X: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(X)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot == 0:
            return 0.0
        return 1.0 - ss_res / ss_tot


def train_predictor_from_library(
    library: ModularElementLibrary,
    n_samples: int = 240,
    random_state: int = 42,
) -> Tuple[RidgeExpressionPredictor, Dict[str, float]]:
    """训练模型并返回验证指标。"""
    dataset = library.generate_training_set(n_samples=n_samples, random_state=random_state)

    X = np.vstack(
        [
            library.build_design_features(
                {
                    "promoter": row.promoter,
                    "rbs": row.rbs,
                    "cds": row.cds,
                    "terminator": row.terminator,
                }
            )
            for row in dataset.itertuples(index=False)
        ]
    )
    y = dataset["predicted_target"].values.astype(float)

    split_index = int(0.8 * len(dataset))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    model = RidgeExpressionPredictor(alpha=0.15)
    model.fit(X_train, y_train)

    metrics = {
        "train_r2": model.score_r2(X_train, y_train),
        "test_r2": model.score_r2(X_test, y_test),
        "samples": float(len(dataset)),
    }
    return model, metrics


def predict_design_expression(
    model: RidgeExpressionPredictor,
    library: ModularElementLibrary,
    design: Dict[str, str],
) -> float:
    x = library.build_design_features(design).reshape(1, -1)
    pred = float(model.predict(x)[0])
    return max(0.01, pred)


def simulate_design_in_circuit(
    circuit,
    design: Dict[str, str],
    library: ModularElementLibrary,
    target_gene: str = "gvpA",
    t_end: float = 100.0,
    t_points: int = 180,
) -> Dict[str, np.ndarray]:
    """将元件组合映射到参数并执行动力学仿真。"""
    promoter = library.get_element("promoter", design["promoter"])
    rbs = library.get_element("rbs", design["rbs"])
    cds = library.get_element("cds", design["cds"])
    terminator = library.get_element("terminator", design["terminator"])

    original_alpha = circuit.params["alpha"][target_gene]
    original_beta = circuit.params["beta"][target_gene]
    original_gamma = circuit.params["gamma"][target_gene]

    try:
        circuit.set_parameter("alpha", target_gene, original_alpha * promoter.strength)
        circuit.set_parameter("beta", target_gene, original_beta * 0.65 * (rbs.strength + cds.strength))
        # 终止子强度越高，等价为更强的表达稳定性（较低有效降解）
        adjusted_gamma = max(0.03, original_gamma / max(0.6, terminator.strength))
        circuit.set_parameter("gamma", target_gene, adjusted_gamma)

        t_eval = np.linspace(0, t_end, t_points)
        sim = circuit.simulate(t_span=(0, t_end), t_eval=t_eval)
        return {
            "t": sim["t"],
            "protein": sim["protein"][target_gene],
            "mRNA": sim["mRNA"][target_gene],
            "final_protein": sim["protein"][target_gene][-1],
        }
    finally:
        circuit.set_parameter("alpha", target_gene, original_alpha)
        circuit.set_parameter("beta", target_gene, original_beta)
        circuit.set_parameter("gamma", target_gene, original_gamma)
