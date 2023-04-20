# syntax=docker/dockerfile:1

FROM tensorflow/tensorflow:2.8.1 AS base

# In [4]: tf.__version__
# Out[4]: '2.8.1'

# In [3]: tensorboard.__version__
# Out[3]: '2.8.0'

WORKDIR /viz

RUN apt-get update && apt-get install nano -y

RUN python -m pip install --upgrade pip
RUN pip install ipython
RUN pip install pandas==1.1.5
