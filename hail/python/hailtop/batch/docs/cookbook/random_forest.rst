.. _sec-cookbook-random_forest:

===================
Random Forest Model
===================

Introduction
------------

We want to use a random forest model to predict regional mutability of
the genome (at a scale of 50kb) using a series of genomic features. Specifically,
we divide the genome into non-overlapping 50kb windows and we regress
the observed/expected variant count ratio (which indicates the mutability
of a specific window) against a number of genomic features measured on each
corresponding window (such as replication timing, recombination rate, and
various histone marks). For each window under investigation, we fit the
model using all the rest of the windows and then apply the model to
that window to predict its mutability as a function of its genomic features.

To perform this analysis with Batch, we will first use a :class:`.PythonJob`
to execute a Python function directly for each window of interest. Next,
we will add a mechanism for checkpointing files as the number of windows
of interest is quite large (~52,000). Lastly, we will add a mechanism to batch windows
into groups of 10 to amortize the amount of time spent copying input
and output files compared to the time of the actual computation per window
(~30 seconds).


Batch Code
----------

~~~~~~~
Imports
~~~~~~~

We import all the modules we will need. The random forest model code comes
from the `sklearn` package.

.. code-block:: python

    import hailtop.batch as hb
    import hailtop.fs as hfs
    from hailtop.utils import grouped
    import pandas as pd
    from typing import List, Optional, Tuple
    import argparse
    import sklearn


~~~~~~~~~~~~~~~~~~~~~~
Random Forest Function
~~~~~~~~~~~~~~~~~~~~~~

The inputs to the random forest function are two data frame files. `df_x`
is the path to a file containing a Pandas data frame where the variables
in the data frame represent the number of genomic features measured on each
corresponding window. `df_y` is the path to a file containing a Pandas data
frame where the variables in the data frame are the observed and expected variant
count ratio.

We write a function that runs the random forest model and leaves the window
of interest out of the model `window_name`.

An important thing to note in the code below is the number of cores is a parameter
to the function and matches the number of cores we give the job in the Batch control
code below.


.. code-block:: python
    :emphasize-lines: 15

    def random_forest(df_x_path: str, df_y_path: str, window_name: str, cores: int = 1) -> Tuple[str, float, float]:
        # read in data
        df_x = pd.read_table(df_x_path, header=0, index_col=0)
        df_y = pd.read_table(df_y_path, header=0, index_col=0)

        # split training and testing data for the current window
        x_train = df_x[df_x.index != window_name]
        x_test = df_x[df_x.index == window_name]

        y_train = df_y[df_y.index != window_name]
        y_test = df_y[df_y.index == window_name]

        # run random forest
        rf = RandomForestRegressor(n_estimators=100,
                                   n_jobs=cores,
                                   max_features=3/4,
                                   oob_score=True,
                                   verbose=False)

        rf.fit(x_train, y_train)

        # apply the trained random forest on testing data
        y_pred = rf.predict(x_test)

        # store obs and pred values for this window
        obs = y_test["oe"].to_list()[0]
        pred = y_pred[0]

        return (window_name, obs, pred)


~~~~~~~~~~~~~~~~~~~~~~
Format Result Function
~~~~~~~~~~~~~~~~~~~~~~

The function below takes the expected output of the function `random_forest`
and returns a tab-delimited string that will be used later on when concatenating results.

.. code-block:: python

    def as_tsv(input: Tuple[str, float, float]) -> str:
        return '\t'.join(str(i) for i in input)


~~~~~~~~~~~~~~~~~~
Build Python Image
~~~~~~~~~~~~~~~~~~

In order to run a :class:`.PythonJob`, Batch needs an image that has the
same version of Python as the version of Python running on your computer
and the Python package `dill` installed. Batch will automatically
choose a suitable image for you if your Python version is 3.8 or newer.
You can supply your own image that meets the requirements listed above to the
method :meth:`.PythonJob.image` or as the argument `default_python_image` when
constructing a Batch . We also provide a convenience function :func:`.docker.build_python_image`
for building an image that has the correct version of Python and `dill` installed
along with any desired Python packages.

For running the random forest, we need both the `sklearn` and `pandas` Python
packages installed in the image. We use :func:`.docker.build_python_image` to build
an image and push it automatically to the location specified (ex: `us-docker.pkg.dev/hail-vdc/random-forest`).

.. code-block:: python

    image = hb.build_python_image('us-docker.pkg.dev/hail-vdc/random-forest',
                                  requirements=['sklearn', 'pandas'])

~~~~~~~~~~~~
Control Code
~~~~~~~~~~~~

We start by defining a backend.

.. code-block:: python

    backend = hb.ServiceBackend()


Second, we create a :class:`.Batch` and specify the default Python image to
use for Python jobs with `default_python_image`. `image` is the return value
from building the Python image above and is the full name of where the newly
built image was pushed to.

.. code-block:: python

    b = hb.Batch(name='rf',
                 default_python_image=image)


Next, we read the `y` dataframe locally in order to get the list of windows
to run. The file path containing the dataframe could be stored on the cloud.
Therefore, we use the function `hfs.open` to read the data regardless
of where it's located.

.. code-block:: python

    with hfs.open(df_y_path) as f:
        local_df_y = pd.read_table(f, header=0, index_col=0)


Now we have a `y` dataframe object on our local computer, but we need a way
for Batch to localize the files as inputs to a :class:`.Job`. Therefore, we
use the method :meth:`.Batch.read_input` to tell Batch to localize these files
when they are referenced by a :class:`.Job`.

.. code-block:: python

    df_x_input = b.read_input(df_x_path)
    df_y_input = b.read_input(df_y_path)


We initialize a list to keep track of all of the output files to concatenate
later on.

.. code-block:: python

    results = []


We now have all of our inputs ready and can iterate through each window in the
`y` dataframe. For each window, we create a new :class:`.PythonJob` using the
method :meth:`.Batch.new_python_job`. We then use the method :meth:`.PythonJob.call`
to run the function `random_forest`. The inputs to `random_forest` are the Batch inputs
`df_x_input` and `df_y_input` as well as the window name. Notice that the first argument to
:meth:`.PythonJob.call` is the reference to the function to call (i.e `random_forest` and
not `random_forest(...)`. The rest of the arguments are the usual positional arguments and
key-word arguments to the function. Lastly, we assign the result of calling the function
to the variable `result` which is a :class:`.PythonResult`. A :class:`.PythonResult`
can be thought of as a Python object and used in subsequent calls to :meth:`.PythonJob.call`.

Since the type of `result` is a tuple of (str, float, float), we need to convert the Python
tuple to a tab-delimited string that can later be concatenated. We use the `as_tsv` function
we wrote above to do so. The input to `as_tsv` is `result` and we assign the output to `tsv_result`.

Lastly in the for loop for each window, we append the `tsv_result` to the results list. However,
`tsv_result` is a Python object. We use the :meth:`.PythonResult.as_str` method to convert the
Python object to a text file containing the `str()` output of the Python object.

.. code-block:: python

    for window in local_df_y.index.to_list():
        j = b.new_python_job()

        result = j.call(random_forest, df_x_input, df_y_input, window)
        tsv_result = j.call(as_tsv, result)
        results.append(tsv_result.as_str())


Now that we have computed the random forest results for each window, we can concatenate
the outputs together into a single file using the :func:`.concatenate` function and then
write the concatenated results file to a permanent output location.

.. code-block:: python

    output = hb.concatenate(b, results)
    b.write_output(output, results_path)

Finally, we call :meth:`.Batch.run` to execute the batch and then close the backend.

.. code-block:: python

    b.run(wait=False)
    backend.close()


Add Checkpointing
-----------------

The pipeline we wrote above is not resilient to failing jobs. Therefore, we can add
a way to checkpoint the results so we only run jobs that haven't already succeeded
in future runs of the pipeline. The way we do this is by having Batch write the computed
result to a file and using the function `hfs.exists` to check whether the file already
exists before adding that job to the DAG.

First, we define the checkpoint path for each window.

.. code-block:: python

    def checkpoint_path(window):
        return f'gs://my_bucket/checkpoints/random-forest/{window}'


Next, we define the list of results we'll append to:

.. code-block:: python

    results = []


Now, we take our for loop over the windows from before, but now we check whether
the checkpointed already exists. If it does exist, then we read the checkpointed
file as a :class:`.InputResourceFile` using :meth:`.Batch.read_input` and append
the input to the results list. If the checkpoint doesn't exist and we add the job
to the batch, then we need to write the results file to the checkpoint location
using :meth:`.Batch.write_output`.

.. code-block:: python

    for window in local_df_y.index.to_list():
        checkpoint = checkpoint_path(window)
        if hfs.exists(checkpoint):
            result = b.read_input(checkpoint)
            results.append(result)
            continue

        j = b.new_python_job()

        result = j.call(random_forest, df_x_input, df_y_input, window)
        tsv_result = j.call(as_tsv, result)
        tsv_result = tsv_result.as_str()

        b.write_output(tsv_result, checkpoint)
        results.append(tsv_result)



Add Batching of Jobs
--------------------

If we have a lot of short running jobs, then we might want to run the calls
for multiple windows at a time in a single job along with the checkpointing
mechanism to check if the result for the window has already completed.

Building on the solution above for checkpointing, we need two for loops
instead of one to ensure we still get an even number of jobs in each
batch while not rerunning previously completed windows.

First, we create a results array that is the size of the number of windows

.. code-block:: python

    indices = local_df_y.index.to_list()
    results = [None] * len(indices)

We identify all of the windows whose checkpoint file already exists
and append the inputs to the results list in the correct position in the
list to ensure the ordering of results is consistent. We also create
a list that holds tuples of the window to compute, the index of that
window, and the checkpoint path.

.. code-block:: python

    inputs = []

    for i, window in enumerate(indices):
        checkpoint = checkpoint_path(window)
        if hfs.exists(checkpoint):
            result = b.read_input(checkpoint)
            results[i] = result
            continue

        inputs.append((window, i, checkpoint))


Then we have another for loop that uses the `hailtop.grouped`
function to group the inputs into groups of 10 and create a
job for each group. Then we create a :class:`.PythonJob` and
use :meth:`.PythonJob.call` to run the random forest function
for each window in that group. Lastly, we append the result
to the correct place in the results list.

.. code-block:: python

    for inputs in grouped(10, inputs):
        j = b.new_python_job()
        for window, i, checkpoint in inputs:
            result = j.call(random_forest, df_x_input, df_y_input, window)
            tsv_result = j.call(as_tsv, result)
            tsv_result = tsv_result.as_str()

            b.write_output(tsv_result, checkpoint)
            results[i] = tsv_result



Now we've only run the jobs in groups of 10 for jobs that have no
existing checkpoint file. The results will be concatenated in the correct
order.


Synopsis
--------

We have presented three different ways with increasing complexity to write
a pipeline that runs a random forest for various windows in the genome. The
complete code is provided here for your reference.

.. literalinclude:: files/run_rf_simple.py
    :language: python
    :caption: run_rf_simple.py

.. literalinclude:: files/run_rf_checkpoint.py
    :language: python
    :caption: run_rf_checkpoint.py

.. literalinclude:: files/run_rf_checkpoint_batching.py
    :language: python
    :caption: run_rf_checkpoint_batching.py
