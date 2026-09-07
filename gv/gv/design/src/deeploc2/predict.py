def predict(sequences, device="cpu", **kwargs):
    """Mock DeepLoc2 prediction - 统一接口"""
    if isinstance(sequences, str):
        sequences = [sequences]

    results = []
    for seq in sequences:
        # 简单的启发式：基于疏水性估计亚细胞定位
        hydrophobic = set('AILMFWVY')
        h_ratio = sum(1 for aa in seq if aa in hydrophobic) / len(seq) if seq else 0

        if h_ratio > 0.55:
            loc = ["Membrane"]
            membrane = "Membrane"
            soluble = 0.3
        elif h_ratio > 0.4:
            loc = ["Cytoplasm", "Membrane"]
            membrane = "Soluble"
            soluble = 0.7
        else:
            loc = ["Cytoplasm"]
            membrane = "Soluble"
            soluble = 0.9

        results.append({
            "localizations": loc,
            "signals": [],
            "membrane": membrane,
            "soluble": soluble,
        })

    # 返回DataFrame格式以兼容DeepLoc-2
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        return df
    except ImportError:
        return results