# Acquire a dataset by what comes back

A handler persists files in a data store and loads them for a source. A fetcher returns the result in memory and imposes no data-store layout. Publication as files or as a query does not decide which one it is. A handler is not defined as a cache over a fetcher: the two can be independent publications.

Classifying by publication form was rejected, because a fetcher can read stable remote files in place and a file payload can be opened in memory or saved. Defining a handler as a wrapper over a fetcher was rejected, because that fits one product and other pairs are separate publications.
