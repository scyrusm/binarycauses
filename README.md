# QUICKSTART:
Either (in a virtualenv)
```pip -r requirements.txt
python binary_causes.py```
Or
`python setup.py install
python binary_causes.py```


# Background
The idea behind this module is to develop a simple algorithm that can detect unary causal relationships (i.e., A causes B) and binary causal relationships (A and B act together to cause C).  The justification for this is that, in practice, many biological causal effects will be of this form; while nothing about graphical causal models (see Judea Pearl's "Causality: Models, Reasoning and Inference") demands that causal effects be unary, in practice, all algorithms rely on some conditional independence tests to resolve a causal graph from data, and those tests (most commonly the Fisher Z test) rely on unary interactions with Gaussian noise.  

The following script is a partial-implementation of PC algorithm (see Spirtes, Glymour, Scheines "Prediction, Causation, and Search"), where we check for both unary interaction and binary interactions--we still assume normal variables, and binary interactions are regarded as simply being the product of those variables (which will themselves still be normal, and therefore conditional independence is testable via the Fisher Z test).  For details of the Fisher Z test, see "Causal Search in Structural Vector Autoregressive Models" (Moneta, Chlaß, Entner, Hoyer, 2011)).  Here, the only addition that is made to the PC algorithm is that, for cases where there are two individual unary interactions (causes) influencing the same effect, an additional check is made to see if there is a significant binary effect (since a binary effect, whether holding the co-causer constant or not, should register as a unary effect as well--though I have not checked the robustness of this assumption).  Furthermore, we are interested in something akin to a time series--i.e., n variables which interact causally with the same n variables at a later time.  It is forbidden that these "later time" variables affect the "earlier time" variables, but it is not forbidden that variables effect variables at the same time step.  Note also, that in this prototypical implementation, I am simply assuming three time steps--one start, one hidden, one final.  

Currently the artificial data generation has various hard-coded parameters, whose effect should be tested.  Moreover, the normal variable assumption almost certainly fails in e.g. a genetic constant (where the interesting variables of interest may be mutation i.e. mostly binary).  It might be interesting to test this model in a population dynamics context.   
