# Under construction

# One Transformer for All Time Series: Representing and Training with Time-Dependent Heterogeneous Tabular Data
Official PyTorch implementaton of paper One Transformer for All Time Series: Representing and Training with Time-Dependent Heterogeneous Tabular Data.  

## Introduction
There is a recent growing interest in applying Deep Learning techniques to tabular data in order to replicate the success of other Artificial Intelligence areas in this structured domain. Particularly interesting is the case in which tabular data have a time dependence, such as, for instance, financial transactions. However, the heterogeneity of the tabular values, in which categorical elements are mixed with numerical features, makes this adaptation difficult. 
In this paper we propose UniTTab, a Transformer based architecture whose goal is to uniformly represent heterogeneous time-dependent tabular data, in which both numerical and categorical features are described using continuous embedding vectors. Moreover, differently from common approaches, which use a combination of different loss functions for training with both numerical and categorical targets, UniTTab is uniformly trained with a unique Masked Token pretext task. Finally, UniTTab can also represent time series in which the individual row components have a variable internal structure with a variable number of fields, which is a common situation in many application domains, such as in real world transactional data. 
Using extensive experiments with five datasets of variable size and complexity, we empirically show that UniTTab consistently and significantly improves the prediction accuracy over several downstream tasks and with respect to both Deep Learning and more standard Machine Learning approaches.
