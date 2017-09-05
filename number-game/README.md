# Number Game
This framework includes two players, a sender and a receiver.

## Game 1
In this game, the sender is given two one-hot vectors and a target(left/right): (vL, vR, t), t \in {L, R}.

The sender need to send a real number to the receiver, we call this the sender's policy: s(vL, vR, t).

The receiver does not know the target, but sees the sender's value and tries to guess the target, we call this the receiver's policy r(vL, vR, s(vL, vR, t)) \in {L, R}.

If r(vL', vR', s(vL, vR, t)) = t, both players receive a payoff of 1, otherwise, they receive a payoff of 0.

## Game 2
The setting of game 2 is the same as game 1, but in game 2, vt = max(vL, vR).

# Algorithm
We adopt the same algorithm as the one in [Multi-Agent Cooperation and the Emergence of (Natural) Language](https://arxiv.org/pdf/1612.07182.pdf).

Notice that the sender's policy is a real number, Deterministic Policy Gradient is required to learn the policy.