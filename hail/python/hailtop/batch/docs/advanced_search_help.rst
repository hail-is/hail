.. _sec-advanced_search_help:

====================
Advanced Search Help
====================

A query has one statement per line. A statement is one of the expressions (exact match,
partial match, keyword, or predefined keyword) listed below. When the query is run, each
statement will be joined to the next with the ``AND`` operator.

Exact Match Expression
----------------------

A single word enclosed with double quotes that is an exact match for either the name or
value of an attribute.

**Example:** ``"pca_pipeline"``

Partial Match Expression
------------------------

A single word without any quotes that is a partial match for either the name or the value
of an attribute.

**Example:** ``pipe``

Keyword Expression
------------------

The left hand side of the statement is the name of the attribute and the right hand side
is the value to search against. Allowed operators are ``=``, ``==``, ``!=``, ``=~``, and
``!~`` where the operators with tildes are looking for partial matches.

**Example:** ``name = pca_pipeline``

**Example:** ``name =~ pca``

Predefined Keyword Expression
-----------------------------

The left hand side of the statement is a special Batch-specific keyword which can be one of the values
listed in the tables below. Allowed operators are dependent on the type of the value expected for each
keyword, but can be one of ``=``, ``==``, ``!=``, ``>``, ``>=``, ``<``, ``<=``, ``=~``, ``!~``.
The right hand side is the value to search against.

.. list-table:: Keywords
    :widths: 25 25 50 50
    :header-rows: 1

    * - Keyword
      - Value Type
      - Allowed Operators
      - Extra
    * - cost
      - float
      - ``=``, ``==``, ``!=``, ``>``, ``>=``, ``<``, ``<=``
      -
    * - duration
      - float
      - ``=``, ``==``, ``!=``, ``>``, ``>=``, ``<``, ``<=``
      - Values are rounded to the millisecond
    * - start_time
      - date
      - ``=``, ``==``, ``!=``, ``>``, ``>=``, ``<``, ``<=``
      - ISO-8601 datetime string
    * - end_time
      - date
      - ``=``, ``==``, ``!=``, ``>``, ``>=``, ``<``, ``<=``
      - ISO-8601 datetime string


**Example:** ``cost >= 1.00``

**Example:** ``duration > 5``

**Example:** ``start_time >= 2023-02-24T17:15:25Z``


.. list-table:: Keywords specific to searching for batches
    :widths: 25 25 50 50
    :header-rows: 1

    * - Keyword
      - Value Type
      - Allowed Operators
      - Extra
    * - batch_id
      - int
      - ``=``, ``==``, ``!=``, ``>``, ``>=``, ``<``, ``<=``
      -
    * - state
      - str
      - ``=``, ``==``, ``!=``
      - Allowed values are `running`, `complete`, `success`, `failure`, `cancelled`, `open`, `closed`
    * - user
      - str
      - ``=``, ``==``, ``!=``, ``=~``, ``!~``
      -
    * - billing_project
      - str
      - ``=``, ``==``, ``!=``, ``=~``, ``!~``
      -


**Example:** ``state = running``

**Example:** ``user = johndoe``

**Example:** ``billing_project = johndoe-trial``



.. list-table:: Keywords specific to searching for jobs in a batch
    :widths: 25 25 50 50
    :header-rows: 1

    * - Keyword
      - Value Type
      - Allowed Operators
      - Extra
    * - job_id
      - int
      - ``=``, ``==``, ``!=``, ``>``, ``>=``, ``<``, ``<=``
      -
    * - state
      - str
      - ``=``, ``==``, ``!=``
      - Allowed values are `pending`, `ready`, `creating`, `running`, `live`, `cancelled`, `error`, `failed`, `bad`, `success`, `done`
    * - instance
      - str
      - ``=``, ``==``, ``!=``, ``=~``, ``!~``
      - use this to search for all jobs that ran on a given worker
    * - instance_collection
      - str
      - ``=``, ``==``, ``!=``, ``=~``, ``!~``
      - use this to search for all jobs in a given pool


**Example:** ``user = johndoe``

**Example:** ``billing_project = johndoe-trial``

**Example:** ``instance_collection = standard``


Combining Multiple Statements
-----------------------------

**Example:** Searching for batches in a time window

.. code-block::

    start_time >= 2023-02-24T17:15:25Z
    end_time <= 2023-07-01T12:35:00Z

**Example:** Searching for batches that have run since June 2023 that cost more than $5 submitted by a given user

.. code-block::

    start_time >= 2023-06-01
    cost > 5.00
    user = johndoe

**Example:** Searching for failed batches where the batch name contains pca

.. code-block::

    state = failed
    name =~ pca

**Example:** Searching for jobs within a given range of ids

.. code-block::

    job_id >= 1000
    job_id < 2000
