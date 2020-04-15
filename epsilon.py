from numpy import exp, ceil


def linear(episode: int, epsilon: float) -> float:
    return epsilon - 0.1 * episode


def exponent(episode: int, epsilon: float) -> float:
    return epsilon - 0.000045 * exp(episode)


def slider(episode: int, epsilon: float) -> float:
    slide = ceil(episode / 2) * 2
    return epsilon - 0.1 * slide
