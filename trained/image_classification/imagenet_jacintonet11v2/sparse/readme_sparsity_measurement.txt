Important information:
======================
The sparsity reported in run.log is not correct as the denominator included fully connected weights as well.
(Fully connected weights contributes to much less compute/MACs compared to convolutional layers - so the fully connected layers are not to be included in sparsity computation).
The correct sparsity for this model is reported in ../test/run.log.
