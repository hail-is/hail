=====================
Other Cloud Providers
=====================

Amazon Web Services
-------------------

While Hail does not have any built-in tools for working with
`Amazon EMR <https://aws.amazon.com/emr/>`__, we recommend the `open-source
tool <https://github.com/hms-dbmi/hail-on-AWS-spot-instances>`__ developed by Carlos De Niz
with the `Avillach Lab <https://avillach-lab.hms.harvard.edu/>`_ at Harvard Medical School

Microsoft Azure
---------------

The step by step, latest process documentation for creating a hail-capable cluster in 
Azure, utilizing an HDInsight Spark Cluster can be found 
`here <https://github.com/TheEagleByte/azure-hail>`__ compiled by Garrett Bromley with 
`E360 Genomics at IQVIA. <https://www.iqvia.com/solutions/real-world-evidence/platforms/e360-real-world-data-platform>`__

Databricks
----------

Hail can be installed on a Databricks Spark cluster on Microsoft Azure, Amazon Web Services, or Google Cloud Platform 
via an open source Docker container located `here <https://hub.docker.com/r/projectglow/databricks-hail/tags?page=1&ordering=last_updated>`__. 
Docker files to build your own Hail container can be found 
`here <https://github.com/projectglow/glow/tree/master/docker>`__.
And further guidelines about working with Hail can be found on the `Databricks documentation <https://docs.databricks.com/applications/genomics/genomics-libraries/hail.html>`__. 

Others
------

There are no known open-source resources for working with Hail on other cloud
providers. If you know of one, please submit a pull request to add it here!

If you have scripts for working with Hail on other cloud providers, we may be
interested in including those scripts in ``hailctl`` (see above) as new
modules. Stop by the `dev forum <https://dev.hail.is>`__ to chat!
