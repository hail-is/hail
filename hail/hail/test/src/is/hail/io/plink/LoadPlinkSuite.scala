package is.hail.io.plink

import is.hail.TestUtils._
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.io.SkipInvalidLociResources
import is.hail.types.physical.LocusRepresentations.forEachCanonicalLocus
import is.hail.variant.{Locus, ReferenceGenome}
import is.hail.variant.BuiltinLoci.builtin

import scala.collection.immutable.ArraySeq

import org.apache.spark.sql.Row
import org.json4s.{DefaultFormats, Extraction}
import org.junit.jupiter.api.Test

class LoadPlinkSuite {
  private def reader(
    name: String,
    contigRecoding: Map[String, String] = Map.empty,
    skipInvalidLoci: Boolean = false,
  )(implicit ctx: ExecuteContext
  ): MatrixPLINKReader = {
    val params = MatrixPLINKReaderParameters(
      bed = getTestResource(s"$name.bed"),
      bim = getTestResource(s"$name.bim"),
      fam = getTestResource(s"$name.fam"),
      nPartitions = None,
      blockSizeInMB = None,
      minPartitions = None,
      delimiter = "\\\\s+",
      missing = "NA",
      quantPheno = false,
      a2Reference = true,
      rg = Some(ReferenceGenome.GRCh37),
      contigRecoding = contigRecoding,
      skipInvalidLoci = skipInvalidLoci,
    )
    MatrixPLINKReader.fromJValue(ctx, Extraction.decompose(params)(DefaultFormats))
  }

  @Test def testRowLocusTypeIsCanonical(implicit ctx: ExecuteContext): Unit = {
    val grch37 = builtin(ReferenceGenome.GRCh37)
    val results = ArraySeq.newBuilder[IndexedSeq[Row]]
    forEachCanonicalLocus { canonicalLocusType =>
      val r = reader("balding-nichols-1024-variants-4-samples-3-populations")
      val (pt, rows) = readRowsLocally(r)
      assertEq(pt.fieldType("locus").setRequired(false), canonicalLocusType(grch37, false))
      assertEq(rows.length, 1024)
      assertEq(
        Row.fromSeq(rows.head.toSeq.take(3)),
        Row(Locus("1", 1), FastSeq("A", "C"), "1:1:A:C"),
      )
      assertEq(rows.last.getAs[Locus](0), Locus("1", 1024))
      results += rows
    }
    val ArraySeq(struct, packed) = results.result(): @unchecked
    assertEq(packed, struct)
  }

  @Test def testSkipInvalidLoci(implicit ctx: ExecuteContext): Unit = {
    val grch37 = builtin(ReferenceGenome.GRCh37)
    val recoding = Map("chr1" -> "1")
    forEachCanonicalLocus { canonicalLocusType =>
      val r = reader("skip_invalid_loci", recoding, skipInvalidLoci = true)
      val (pt, rows) = readRowsLocally(r, "locus", "alleles")
      assertEq(pt.fieldType("locus").setRequired(false), canonicalLocusType(grch37, false))
      assertEq(rows, SkipInvalidLociResources.validRows)

      interceptFatal("Invalid locus 'chr0:1'") {
        readRowsLocally(reader("skip_invalid_loci", recoding), "locus", "alleles")
      }
    }
  }
}
