def rank_slots(slots, energy_level):
    weights = {"low": 0.5, "medium": 1.0, "high": 1.5}
    return sorted(slots, key=lambda s: s["availability"] * weights.get(energy_level, 1))