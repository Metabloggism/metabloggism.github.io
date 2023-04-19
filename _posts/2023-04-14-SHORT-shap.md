# **Short: Shap values**

Hello readers, it has been quite some time since our last blog post, and for that, I apologize. Due to personal and work matters, I have been away for almost two months. However, I am excited to be back and share with you a topic that I think you will find interesting: Shap values.

## **Motivation: Why is this today's topic?**

I recently used Shap values in a project and found it to be a chance to tell you something of interest. Shap values are something I've known for some time and they are a powerful and incredibly useful tool for explainable AI. While many of you may be familiar with Shap values, I will take the time to contextualize where they come from, how they work, and how to use them in your projects. For the people who know what it is, you may want to check what you are actually using, while for those who are unfamiliar, I would recommend you use this as an opportunity to learn a new tool and acquire the criterion to choose when it may be useful. So, let's dive in!

## **Some Game Theory principles**
 
No, I'm not throwing random topics at you. Actually, Game Theory is highly correlated to Shap values and now you'll know why.
 
Game Theory is not a game. It is a branch of mathematics that aims to **understand how people make decisions when their choices are influenced by others**. It helps us understand how different individuals or groups (and attention to that difference) interact strategically in a decision-making scenario. So it provides a powerful framework to model the behavior of rational agents. Game theory has many applications, from economics and politics to biology and psychology, and its concepts have been used to analyze and solve a variety of real-world problems.

In essence, Game theory considers that the outcome of a game depends on the strategies chosen by all players involved. Each player has a set of possible actions, and each action leads to a specific outcome. The players aim to maximize their payoff, whether monetary or non-monetary (but quantizable anyway), by choosing a strategy that gives them the best chance of success. Game theory assumes that players are rational and aware of each other's strategies, and therefore, they act strategically to maximize their outcomes.

Game theory provides several tools/scenarios for analyzing strategic interactions. An example of a Game Theory scenario is Nash equilibrium, where no player has the incentive to change their strategy. 

Another important concept in Game Theory is the idea of cooperative games, where players form coalitions and share the payoff. In cooperative games, the focus is on how to distribute the payoff among the players, which leads us to the concept of Shapley values in Game Theory.

![](https://www.gametheory.online/projects/1547107965.jpg)

## **Shap values in Game Theory**

Shapley values were born as a concept in cooperative Game Theory, aiming to distribute the payoff among the players fairly and efficiently. It was introduced by [Lloyd Shapley](https://en.wikipedia.org/wiki/Lloyd_Shapley) in 1953 and has since become an important tool in game theory and Machine Learning. Shapley values take into account the contributions of each player to the coalition, considering all possible ways that the players can form a coalition. The idea is to calculate the marginal contribution of each player to the coalition, which is the difference between the payoff of the coalition with the player and without the player.

The Shapley values provide a unique way to distribute the payoff among the players, based on their contributions to the coalition. They are widely used in cooperative game theory to solve problems such as resource allocation, cost sharing, and voting systems. 

![](https://miro.medium.com/v2/resize:fit:1400/1*AzGc8wSKrP7TzLh84N8Lcg.png)

## **From Game Theory to predictive modeling**

In machine learning, Shapley values are used to explain the output of a model by assigning a contribution to each feature of the input. This allows us to understand the importance of each feature in the model's decision and to identify any biases or inconsistencies. Shapley values have become an essential tool in the field of explainable AI, providing a way to interpret the output of complex models and ensuring transparency and fairness in their decision-making processes.

An important note on Shap values is that they are strictly detached from the model. This also gives it a Model-Agnostic nature which is a crucial feature to have (in some scenarios I would say unnegotiable).

## **An idea of how are SHAP values computed**

The scope of this post is not to be a tutorial, but in order to get a better idea of Shap values, there are some key ideas on the computation:

* Shapley values are computed by considering all possible permutations of players in a coalition
* The marginal contribution of each player is calculated for each permutation
* The marginal contribution is the difference in the value of the coalition with and without the player, considering all possible subsets of the coalition
* The formula for computing Shapley values involves averaging the marginal contributions over all possible permutations of players in the coalition, which can be time-consuming for larger datasets and complex models
* Efficient algorithms such as the Monte Carlo method and Shapley value regression have been developed to speed up the computation of Shapley values
* The Shapley value regression uses a linear regression model to approximate the Shapley values based on a set of inputs and their associated outcomes

## **Shap values starter pack**

* Python's library *shap* (`pip install shap`)

That's all. Yes, python already offers you a package with all you need. This, with a good tutorial on Shap values usage, should be enough to use in your project.

![](https://shap.readthedocs.io/en/latest/_images/shap_header.png)

## **The "bad" side**

Although Shap values are a simple yet powerful tool for explaining ML predictive models, they still require the user to understand their value or otherwise they will be giving empty insights. 

For example, don't you miss here any mention of Computer Vision, NLP, or other fields? Well, Shap values are especially valuable for exploring the contribution of raw features in a model. However, in such fields, we already know a complex feature structure that the models will learn from. For example, in Computer Vision the most granular features are the image pixels (in a single channel, single frame). However, as any Computer Vision community member knows, to be feasible instead of feeding a linear model with the raw pixels, we use (normally) convolutional models over a set of values with a predefined relation among them. This relation is above any insight that Shap values can give since it comes from the nature of our knowledge, and Shap values will intuitively not give any valuable insight from this kind of feature. In the case of CNN's, there are other ways to interpret and explain models, such as filter visualization or more automatic tools (which deserve a specific topic).

Does that mean (in this case) that Shapley values have no place in Computer Vision? Obviously not, there are other elements in Computer Vision that interact in a more raw manner, such as channels or more concrete elements in custom pipelines (such as instances in some intermediate stages in tasks like captioning, etc).

And the same applies in other fields, such as NLP, where the elements also have their particular relations.

![](https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/gradient_imagenet_plot.png)

## Summarizing

Shapley values are a valuable tool, inspired by cooperative Game theory (where they represent each team member's contribution to the outcome) and adapted to ML predictive modeling in order to explain the contribution of different elements (such as features) to a given prediction. There are tools to compute them efficiently and should be considered for explainable AI, especially where no relation is preset among these elements (e.g. raw columns in tabular data).

## Thank you, reader

Thank you again for reading this post! As you know any feedback is welcome. I really hope we can meet soon this time.

## References

*[1]* **[*Using SHAP Values to Explain How Your Machine Learning Model Works*](https://www.kaggle.com/code/dansbecker/shap-values)**

*[2]* **[*Explainable AI: Application of shapely values in Marketing Analytics*](https://towardsdatascience.com/explainable-ai-application-of-shapely-values-in-marketing-analytics-57b716fc9d1f)**

*[3]* **[*How to interpret machine learning models with SHAP values*](https://dev.to/mage_ai/how-to-interpret-machine-learning-models-with-shap-values-54jf#:~:text=SHAP%20values%20can%20be%20used,to%20explain%20limited%20model%20types.)**

*[4]* **[*Game Theory*](https://plato.stanford.edu/entries/game-theory/)**

*[5]* **[*Understanding Game Theory*](https://www.skillsyouneed.com/lead/game-theory.html)**

*[6]* **[*Explainable Machine Learning, Game Theory, and Shapley Values: A technical review*](https://www.statcan.gc.ca/en/data-science/network/explainable-learning)**

*[7]* **[*Deep Learning Model Interpretation Using SHAP*](https://towardsdatascience.com/deep-learning-model-interpretation-using-shap-a21786e91d16)**


## Some resources

*[1]* **[*Kaggle Shap tutorial*](https://www.kaggle.com/code/dansbecker/shap-values)**

*[2]* **[*SHAP library documentation*](https://shap.readthedocs.io/en/latest/index.html)**
