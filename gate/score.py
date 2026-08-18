from gate.types import Score, SignalResult


def aggregate(results: list[SignalResult]) -> Score:
    """Combine signal suspicions into a trust score via noisy-OR.

    Noisy-OR rather than a mean: signals are independent evidence, so mild
    suspicion from several should compound, and one confident detection must
    not be diluted by calm neighbours.
    """
    clean = 1.0
    for result in results:
        clean *= 1.0 - result.score
    top = max(results, key=lambda r: r.score) if results else None
    return Score(trust=clean, top=top, results=list(results))
