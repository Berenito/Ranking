import pandas as pd
import numpy as np
from algos.bradley_terry import get_bradley_terry_ratings

def test_get_bradley_terry_ratings():
    """
    Test the Bradley-Terry model on a small example.
    This test is based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4274013/pdf/pone.0115367.pdf page 13 example a)
    """
    
    df = pd.DataFrame(
        {
            "Team_1": ["A", "A", "A", "B", "B", "C"],
            "Team_2": ["B", "C", "D", "C", "D", "D"],
            "Score_1": [8, 8, 10, 7, 3, 4],
            "Score_2": [4, 3, 1, 3, 1, 1],
        }
    )
    ratings = get_bradley_terry_ratings(df, iterations=10000)
    win_probability = lambda t1, t2: ratings[t1]/(ratings[t2]+ratings[t1])

    print(win_probability("A", "B"))
    print(win_probability("A", "C"))
    print(win_probability("A", "D"))
    print(win_probability("B", "C"))
    print(win_probability("B", "D"))
    print(win_probability("C", "D"))

    assert np.isclose(win_probability("A", "B"), 0.640, atol=0.01)
    assert np.isclose(win_probability("A", "C"), 0.758, atol=0.01)
    assert np.isclose(win_probability("A", "D"), 0.902, atol=0.01)
    assert np.isclose(win_probability("B", "C"), 0.638, atol=0.01)
    assert np.isclose(win_probability("B", "D"), 0.838, atol=0.01)
    assert np.isclose(win_probability("C", "D"), 0.746, atol=0.01)


