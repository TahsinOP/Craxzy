---
title: "The Central Idea"
description: "Catching is prediction under a deadline. Physics does most of it exactly; a network learns only what physics leaves out, and says how sure it is."
sidebar:
  order: 1
---

:::note[Guiding question]
What has to be known, and how early, for an arm to be in the right place when a thrown object arrives?
:::

## Catching is prediction under a deadline

The arm cannot react to where the object is. By the time a hand has moved 30 cm the object has moved two metres. It has to act on where the object **will be**, and it has to commit early: with the acceleration limits of a Panda, the elbow needs about 0.65 s to travel one radian and stop, which is most of the flight.

So the quantity that matters is the **catch point**: the place where the predicted path crosses a fixed plane in front of the arm, together with the time of that crossing. Everything upstream exists to estimate it, and everything downstream consumes it.

Two properties of that estimate drive the design:

1. **Its error shrinks as the object approaches**, because there is less future left to predict. The arm should therefore start moving on a rough estimate and refine its motion as better ones arrive.
2. **Its error is very different for different objects.** A ball is a point mass with drag. A card is a wing that tumbles.

<figure class="plot">
<img class="fig-light" src="/fig/flights-light.svg" alt="Side view of twelve ball throws and twelve card throws, with the path predicted by the drag-ballistic model from the true launch state">
<img class="fig-dark" src="/fig/flights-dark.svg" alt="Side view of twelve ball throws and twelve card throws, with the path predicted by the drag-ballistic model from the true launch state">
<figcaption>Simulated flights against the drag-ballistic model started from the exact launch state. For the ball the two coincide (median miss 0.6 cm at the plane). For the card the model is wrong by a median of 23 cm even with perfect knowledge of the launch.</figcaption>
</figure>

## What "hybrid" means here

A purely learned predictor has to rediscover gravity from data. A purely physical one has no term for flutter. The stack in these pages splits the work along the line where the physics stops being exact:

| layer | classical part | learned part |
|---|---|---|
| prediction | the simulator's integration step with gravity and quadratic drag, in closed form | the acceleration that step leaves out, as a mean and a variance |
| estimation | an extended Kalman filter | its process noise comes from the network's variance |
| control | inverse kinematics, a trajectory QP, computed torque | none |
| scheduling | periodic and covariance-based triggers | a policy trained with PPO that decides when to replan |

The learned parts are small and their job is narrow. The residual network never predicts a position; it predicts an acceleration correction, and the same physics that was exact for the ball integrates it. That is the structure of residual learning for control (Johannink et al.; O'Connell et al., *Neural-Fly*): keep the model, learn its error.

## Why the variance matters as much as the mean

A predictor that is wrong by 5 cm and knows it is useful: the arm can wait, or cover the uncertain region. A predictor that is wrong by 5 cm and reports 5 mm is dangerous, because every layer after it trusts the number.

The network is therefore trained with a Gaussian likelihood and outputs a variance for every future step (Kendall & Gal). Two further corrections, both fitted on held-out data, are needed before that variance becomes a usable covariance of the catch point; they are the subject of [Calibration](/residual/calibration/). The replanning trigger in part VI is built directly on that covariance.

## One rule that everything obeys

There is exactly one discrete physics model in the project, and it is the simulator's. The throw generator aims with it, the filter predicts with it, the residual is defined against it, and the planner integrates joint motion with the same scheme. The reason is on the next pages: the obvious alternatives are wrong by 3 mm and 3 cm on a ball over one flight, which is the entire error budget of the ball catch.

## What was achieved

| milestone | target | result |
|---|---|---|
| scene and throws | a card tunnels without speculative contacts and does not with them | 100 % against 0 % of 600 throws |
| physics + filter | ball error under 1 cm at 200 ms | 0.35 ± 0.00 cm |
| learned residual | better than physics and than pure learning on the card; calibrated | 3.22 cm against 6.65 and 5.10; coverage within 2.3 points |
| interception | ball caught in more than 90 % of throws | 97.3 ± 0.3 % |
| replanning | catch rate against planner calls for three triggers | measured for four |

What did not work is stated with equal weight in [Limits](/evaluation/limits/).
