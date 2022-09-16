# Offline-aware Knowledge Tracing

ICTAI'2022: Modeling Offline Knowledge Evolution Effect for Online Knowledge Tracing (Pytorch implementation for OKT).

## How to use the code

1. Prepare the dataset you want by running

    ```shell
    python prepare_data/prepare_[dataset].py
    ```
    , where `[prepare_dataset]` can be assist2012, assist2017 and nips2020_1_2.

2. Use the model by running

    ```shell
    python main.py [dataset]
    ```
    , where the optional values of `[dataset]` is the same as above, and if you do not specify the dataset it will default to assist2017.
