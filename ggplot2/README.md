# ggplot redesign

## architecture


## misc

the overall design of the module is purely functional and uses dataclasses as
simple wrappers for data, without functionality unto themselves (except maybe
typechecking?). this is mostly easy to accomplish, with the notable exception of
needing to override the `+` operator in order to mimic the R ggplot interface.
since this must be done via the `__add__` method in python, we compromise by
creating the plot class without it, and adding the method to it downstream.

we take the data argument purely to comply with the R ggplot interface. if we
want to allow the use of things like `hail.if_else`, we need to take in
aesthetic mappings that are expressions unto themselves, so we don't actually
need a reference to the table, since the expressions already reference the
table, wheareas in R, the ggplot function quotes its arguments and later
interprets them as fields over the given data.

types are exposed in case users want to use type annotations, but the
dataclasses are simply wrappers around the data with no inbuilt functionality,
and the user is expected to use the custom constructors like ggplot to create
instances of the classes instead of the dataclass constructors.

we store a simple undo history in order to make iterating with different plot
types easier, and a cache of computations in order to make re-running the same
command faster.

the general idea:

* start with a hail table/matrixtable
* use hail to compute aggregations and/or downsample the data 
* convert the data to a format usable by altair/vega-lite
* draw a graph 

we want to cache as much as possible during this process so that if the user
wants to try something different, we can avoid recomputing. this means caching
the result of each aggregation/downsample and the result of each conversion.

MAYBE we can convert the data to a dictionary in the format that vega-lite
expects instead of a pandas dataframe, and use altair imports that avoid all the
functionality we're skipping over?





NOTES FROM DAN 1:1

each plot object will have a cache of collected data and aggregated data, so
each stat (including identity) just looks at the plot object to find its data

stats could be passed by string, or we can just have objects to make
typechecking easier, each stat should be composed of the smallest sub stats
possible so that we can store the result of each known aggregation and reuse as
desired

when storing agg results we also need to know what columns they're over

ggplot gets mad if there's both an x and y for histogram, so if you want a base
mapping on the plot object and then add a point and a histogram it gets mad, we
could just assume they want the x value unless specified

find the sweet spot of data size where it makes sense to collect and transform
into dict shape that vega wants, above a certain size instead put it in
a dataframe

cache each individual aesthetic's collected data, so if we want x and y and
later x and z, we first collect x, then y, then look at the cache for x, then z

faceting: maybe group by in hail and separately cache, maybe group after collect

scales will run hail aggs and cache the result much like stats, so that altair
is just reading the data directly

talk to dan about sampling for qqplot and histogram, also ask about a good data
example for the manhattan plot

we can also make the size where we collect vs dataframe configurable, have
a sane default but if the user wants to force one or the other or finds a better
value on their machine they can update it

we should allow the user to define a custom stat so that they can get the
benefits of caching their aggregations for things we haven't predefined






THOUGHTS

add config option to disable typechecking

the user is expected to only use hail and ggplot2 functions, with some knowledge
of the underlying dataframe/dict, altair, and vega-lite, but no need to call any
functions on those explicitly

when adding a geom, typecheck against aes on base plot object, so if geom needs
a numeric y, check that we have a numeric expr type on the aes
