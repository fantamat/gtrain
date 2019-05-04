#gtrain

Project that using abstraction of the model and data to define structures that can be used for learning. Model can bee learned by gtrain function to fit the data properly.
This abstraction allows to implement almost everything you want without need to implementing learning algorithm all the time.
For example it can easily handle various input length witch is not common in other implementations.

##Examples

Examples are included in a folder *example* in a github repository.

##Implementation

There are three parts of the abstraction the model, data, and learning function.

###Model

Abstaction of a model in *TensorFlow*.

###Data

Abstraction of the data for feeding the model within the training algorithm.