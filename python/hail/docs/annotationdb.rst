.. _sec-annotationdb:

===================
Annotation Database
===================

This database contains a curated collection of variant annotations in Hail-friendly format, for use in Hail analysis pipelines. 

Currently, the :py:meth:`~.VariantDataset.annotate_variants_db` VDS method associated with this database works only if you are running Hail on the 
Google Cloud Platform. 

To incorporate these annotations in your own Hail analysis pipeline, select which annotations you would like to query from the 
options below and then copy-and-paste the Hail code generated into your own analysis script.

For example, a simple Hail script to load a VCF into a VDS, annotate the VDS with CADD raw and PHRED scores using this database, 
and inspect the schema could look something like this:

.. code-block:: python
  
  import hail
  from pprint import pprint

  hc = hail.HailContext()

  vds = (
      hc
      .import_vcf('gs://annotationdb/test/sample.vcf')
      .split_multi()
      .annotate_variants_db([
          'va.cadd.RawScore', 
          'va.cadd.PHRED'
      ])
  )

  pprint(vds.variant_schema)

This code would return the following schema:

.. code-block:: python

  Struct{
      rsid: String,
      qual: Double,
      filters: Set[String],     
      info: Struct{
          ...    
      },
      cadd: Struct{
          RawScore: Double,
          PHRED: Double 
      }
  }

-------------

Query Builder
-------------
      
.. raw:: html      

      <div class="container">

        <div class="row">

          <div id="panel-annotations" class="panel panel-default">
            <div class="panel-heading query clearfix">
               <a class="pull-left" role="button" data-toggle="collapse" href="#annotations">
                  <span class="text-expand">Annotations</span>
               </a>
               <div class="btn-group pull-right">
                  <a class="btn btn-default btn-sm" id="annotations-clear">Clear selections</a>
               </div>
            </div>
            <div class="panel-collapse collapse in" id="annotations">
               <div class="panel-body annotations">
                  <div id="tree"></div>
               </div>
            </div>
          </div>

          <div id="panel-query" class="panel panel-default col-4">
            <div class="panel-heading query clearfix">
               <a class="pull-left" role="button" data-toggle="collapse" href="#hail-code">
                  <span class="text-expand">Database Query</span>
               </a>
               <div class="btn-group pull-right">
                  <a class="btn btn-default btn-sm" data-clipboard-target="#hail-copy" id="hail-copy-btn">Copy to clipboard</a>
               </div>
            </div>
            <div class="panel-collapse collapse in" id="hail-code">
               <div class="panel-body hail-code">
                  <span id="template-query-before"><pre>vds = (<br>    hc<br>    .read('my.vds')<br>    .split_multi()<br></pre></span>
                  <span class="hail-code query" id="hail-copy"><pre class="import-function">    .annotate_variants_db([<br>        ...<br>    ])</pre></span>
                  <span id="template-query-after"><pre>)</pre></span>
               </div>
            </div>
          </div>

        </div>

      </div>

-------------

Documentation
-------------

These annotations have been collected from a variety of publications and their accompanying datasets (usually text files). Links to 
the relevant publications and raw data downloads are included where applicable.
   
.. raw:: html

   <div class="panel-group" id="panel-docs">
   </div>

---------------

Important Notes
---------------

VEP annotations
===============

VEP annotations are included in this database under the root :code:`va.vep`. To add VEP annotations, the :py:meth:`~.VariantDataset.annotate_variants_db` 
method runs Hail's :py:meth:`~.VariantDataset.vep` method on your VDS. This means that your cluster must be properly initialized as described in the 
*Running VEP* section in this_ discussion post.

.. warning::

    If you want to add VEP annotations to your VDS, make sure to add the initialization action 
    :code:`gs://hail-common/vep/vep/vep85-init.sh` when starting your cluster.

.. _this: http://discuss.hail.is/t/using-hail-on-the-google-cloud-platform/80

Gene-level annotations
======================

Annotations beginning with :code:`va.gene.` are gene-level annotations that can be used to annotate variants in your VDS. These 
gene-level annotations are stored in the database as keytables keyed by HGNC gene symbols. 

By default, if an annotation beginning with :code:`va.gene.` is given to :py:meth:`~.VariantDataset.annotate_variants_db` and no :code:`gene_key` 
parameter is specified, the function will run VEP and parse the VEP output to define one gene symbol per variant in the VDS.

For each variant, the logic used to extract one gene symbol from the VEP output is as follows:

*  Collect all consequences found in canonical transcripts
*  Designate the most severe consequence in the collection, as defined by this hierarchy (from most severe to least severe):

    - Transcript ablation
    - Splice acceptor variant
    - Splice donor variant
    - Stop gained
    - Frameshift variant
    - Stop lost
    - Start lost
    - Transcript amplification
    - Inframe insertion
    - Missense variant
    - Protein altering variant
    - Incomplete terminal codon variant
    - Stop retained variant
    - Synonymous variant
    - Splice region variant
    - Coding sequence variant
    - Mature miRNA variant
    - 5' UTR variant
    - 3' UTR variant
    - Non-coding transcript exon variant
    - Intron variant
    - NMD transcript variant
    - Non-coding transcript variant
    - Upstream gene variant
    - Downstream gene variant
    - TFBS ablation
    - TFBS amplification
    - TF binding site variant
    - Regulatory region ablation
    - Regulatory region amplification
    - Feature elongation
    - Regulatory region variant
    - Feature truncation
    - Intergenic variant

*  Take the gene symbol from the canonical transcript with the most severe consequence

Though this is the default logic, you may wish to define gene symbols differently. One way to do so while still using the VEP output 
would be to add VEP annotations to your VDS, create a gene symbol variant annotation by parsing through the VEP output however you 
wish, and then pass that annotation to :py:meth:`~.VariantDataset.annotate_variants_db` using the :code:`gene_key` parameter.

Here's an example that uses the gene symbol from the first VEP transcript:

.. code-block:: python

  import hail
  from pprint import pprint

  hc = hail.HailContext()

  vds = (
      hc
      .import_vcf('gs://annotationdb/test/sample.vcf')
      .split_multi()
      .annotate_variants_db('va.vep')
      .annotate_variants_expr('va.my_gene = va.vep.transcript_consequences[0].gene_symbol')
      .annotate_variants_db('va.gene.constraint.pli', gene_key = 'va.my_gene')
  )

  pprint(vds.variant_schema)

This code would return:

.. code-block:: python

  Struct{
      rsid: String,
      qual: Double,
      filters: Set[String],     
      info: Struct{
          ...    
      },
      vep: Struct{
          ...
      },
      my_gene: String,
      gene: Struct{
          constraint: Struct{
              pli: Double
          }
      }
  }
