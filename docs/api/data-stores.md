# Data stores

A data store is where a handler keeps its files. `LocalDataStore` is the default. A client is not a data store.

`read_dataset` and `write_dataset` read and write a file through whichever store you pass.

::: gigaspatial.core.io.local_data_store.LocalDataStore
    options:
      show_source: false

::: gigaspatial.core.io.adls_data_store.ADLSDataStore
    options:
      show_source: false

::: gigaspatial.core.io.snowflake_data_store.SnowflakeDataStore
    options:
      show_source: false

::: gigaspatial.core.io.delta_sharing_data_store.DeltaSharingDataStore
    options:
      show_source: false

::: gigaspatial.core.io.readers.read_dataset
    options:
      show_source: false

::: gigaspatial.core.io.writers.write_dataset
    options:
      show_source: false
