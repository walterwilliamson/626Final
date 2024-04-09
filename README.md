# 626Final
Machine Learning Final Project

Guiding Exapmle/Inspiration:
https://thinkingneuron.com/using-artificial-neural-networks-for-regression-in-python/


Meeting Notes 3/20:
- Have a working one layer model, after removing second layer predictions seems to work

Next Steps:
- Read/learn about keras, especially adding multiple layers (This week)
- Make predictions for the training set to test accuracy (DONE)
- Scale/normalize outputs before prediction, and then unscale/untransform for interpretation (to make sure we don't get negative survival times) (This week)
- Add more features (clustering of cells, etc.), could do more research on particularly informative proteins for example
- Hyperparameter selection process

Meeting Notes 3/27:
- Implemented Cross Validation: tested different levels of epochs for mse
- Added new features

Next Steps:
- Could mess around with number of neurons and other parameters in our model
- Think about randomness in ANN: https://www.researchgate.net/publication/312057617_Randomness_in_Neural_Networks_An_Overview
