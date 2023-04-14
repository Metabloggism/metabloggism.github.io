# **Short: Shap values**

Hello readers, it has been quite some time since our last blog post, and for that, I apologize. Due to personal and work matters, I have been away for almost two months. However, I am excited to be back and share with you a topic that I think you will find interesting: Shap values.

## **Motivation: Why is this today's topic?**

 I recently used Shap values in a project and found it to be an incredible chance to tell you something of interest. Shap values are something I've know for some time and they are a powerful and incredibly useful tool for explainable AI. While many of you may be familiar with Shap values, I will take the time to contextualize where they come from, how they work, and how to use them in your projects. For the people who know what it is, you may want to check what you are actually using, while for those who are unfamiliar, I would recommend you to use this as an opportunity to learn a new tool and acquiere the criterion to choose when it may be useful. So, let's dive in!
 
 ## **Some Game Theory principles**
 
 No, I'm not throwing random topics at you. Actually Game Theory is highly correlated to Shap values and now you'll know why.
 
 Game Theory is not a game. It is a branch of mathematics that aims to **understand how people make decisions when their choices are influenced by others**. It helps us understand how different individuals or groups (aand attention to that difference) interact strategically in a decision-making scenario. So it provides a powerful framework to model the behavior of rational agents. Game theory has many applications, from economics and politics to biology and psychology, and its concepts have been used to analyze and solve a variety of real-world problems.

In essence, Game theory considers that the outcome of a game depends on the strategies chosen by all players involved. Each player has a set of possible actions, and each action leads to a specific outcome. The players aim to maximize their payoff, whether monetary or non-monetary (but quantizable  anyway), by choosing a strategy that gives them the best chance of success. Game theory assumes that players are rational and aware of each other's strategies, and therefore, they act strategically to maximize their outcomes.

Game theory provides several tools for analyzing strategic interactions. An exampñle of a Game Theory scenario is Nash equilibrium, where no player has an incentive to change their strategy. 

Another important concept in Game Theory is the idea of cooperative games, where players form coalitions and share the payoff. In cooperative games, the focus is on how to distribute the payoff among the players, which leads us to the concept of Shapley values in Game Theory.

## **Shap values in Game Theory**

Shapley values were born as a concept in cooperative Game Theory, aiming to distribute the payoff among the players in a fair and efficient way. It was introduced by [Lloyd Shapley](https://en.wikipedia.org/wiki/Lloyd_Shapley) in 1953 and has since become an important tool in game theory and Machine Learning. Shapley values take into account the contributions of each player to the coalition, considering all possible ways that the players can form a coalition. The idea is to calculate the marginal contribution of each player to the coalition, which is the difference between the payoff of the coalition with the player and without the player.

The Shapley values provide a unique way to distribute the payoff among the players, based on their contributions to the coalition. They are widely used in cooperative game theory to solve problems such as resource allocation, cost sharing, and voting systems. 

## **From Game Theory to predictive modelling**

In machine learning, Shapley values are used to explain the output of a model by assigning a contribution to each feature of the input. This allows us to understand the importance of each feature in the model's decision and to identify any biases or inconsistencies. Shapley values have become an essential tool in the field of explainable AI, providing a way to interpret the output of complex models and ensuring transparency and fairness in their decision-making processes.

An important note on Shap values is that they are strictly dettached from the model. This also gives it a Model-Agnostic nature which is a crucial feature to have (in some scenarios I would say unnegotiable).



