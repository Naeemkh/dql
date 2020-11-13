## Deep Double Q-learning

![DQlearning Demo](figures/animation_20201003101019.gif)


In this project, Deep Q-learning and Double Deep Q-learning algorithms are implemented to collect gold nuggets and deliver them to predefined storage. The domain is a 2-dimensional maze. The following figure shows an example of the domain and its features.


<figure class="image">
  <img src="figures/domain_example.png" alt="figures/domain_example.png" width="600">
  <figcaption>Example of the project environment. Cube’s color definition is represented on the right. The agent (red cube) is allowed to normal moves. The color of the agent turns green (not shown in the domain) upon collecting a gold. See the text for more details.</figcaption>
</figure>

The agent learns the optimal policy by the DDQL algorithm. The following table shows the immediate rewards for the agent actions. 


|                             Action                                |  Reward | 
| ----------------------------------------------------------------- | ------- |
| Hitting the borders                                               |   -1.00 |
| Hitting the blocks(walls)                                         |   -1.00 |
| Wandering around                                                  |   -0.05 |
| Entering a cube with gold while the agent has gold                |   -0.05 |
| Entering a cube with gold while the agent does not have the gold  |   +1.00 |
| Entering storage while the agent has gold                         |   +1.00 |
| Entering storage while the agent does not have gold               |   -0.20 |

For more details about the domain, please refere to the following files:
- [Domain API](domain.py)
- [Domain interaction](domain_prep.ipynb)