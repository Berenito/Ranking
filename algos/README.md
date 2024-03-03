# Bradley-Terry Model

## Introduction
The core idea of Bradley-Terry is to model the probability of team A winning
over team B as
$$P(A > B) = \frac{\pi_A}{\pi_B + \pi_B}$$
where $\pi_A$ and $\pi_B$ are the rankings of the respective teams.

This is effectivly the same probability model used in the ELO system.
Contrary to the ELO system which is slow to converge,
the Bradley-Terry model can be used to calculate a rankings after
an arbitrary amount of games played. This fits perfectly to the current mode how
the EUF Season is set up.

The Bradley-Terry model only has two fundamental assumptions that need to be met:

1. The graph of teams must be fully connected. This should generally be guaranteed
by EUF tournaments. Otherwise a lot of ranking algorithms will struggle.
2. Linear Stochastic Transitivity. This assumptions is somewhat intuitive to the
way we think about rankings, although not neccessarily true in reality. But for
the sake of keeping the model as simple as possible, I think we can agree for this
assumption to be true.

The Bradley Terry model is parameterless and guaranteed to converge to a global
optimum. Which is nice as there is no need to fiddle with parameters
and there is only one true ranking in this model.

Another nice thing about the model is, that its results can be statistically
interpreted. You can calculate the p-value for its fit,
i.e. one can easily checking its quality of fit.
You can calculate the win probabilities and odds for every matchup,
which might be interesting for spectators or broadcasting.
And you can even interpret the ranking itself as the odds of winning against the
average team in the EUF registered pool.

## Considerations

In this implementation we consider do not simply a wins and losses on the game level.
Since a game of ultimate consists of a series of independent points,
it is reasonable to consider every point as a win over the other team.
This has several advantages:

* The overall score of all games does have an effect on the ranking.
* Since we only have a small amount of EUF certified games,
the sample size is increased by considerering wins at a lower granularity.
* And the algorithm is a lot more stable,
as comparisons where a team has zero wins against another team are less likely.
This helps with keeping the graph strongly connected.
