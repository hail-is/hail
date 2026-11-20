package is.hail.io

import is.hail.collection.FastSeq
import is.hail.variant.Locus

import org.apache.spark.sql.Row

/** The skip_invalid_loci test resources (.vcf, .bim, .bgen) hold the same five variants, A/C at
  * 0:1, 1:0, 1:3, 1:4 and 1:5 (the .bim names the contigs chr0 and chr1). On GRCh37 the first two
  * are invalid. Contig 0 isn't in GRCh37, so a packed locus couldn't be built for 0:1: readers must
  * skip or reject it before building the locus.
  */
object SkipInvalidLociResources {

  /** The locus and alleles of the variants that remain when invalid loci are skipped. */
  val validRows: IndexedSeq[Row] =
    FastSeq(3, 4, 5).map(pos => Row(Locus("1", pos), FastSeq("A", "C")))
}
