package is.hail.io.vcf

import is.hail.TestUtils._
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir.MatrixRead
import is.hail.expr.ir.defs.{I32, MakeStruct, ReadPartition, Str}
import is.hail.io.SkipInvalidLociResources
import is.hail.types.physical.{PCanonicalString, PCanonicalStruct, PInt32}
import is.hail.types.physical.LocusRepresentations.forEachCanonicalLocus
import is.hail.types.virtual.{MatrixType, TFloat64}
import is.hail.utils.{HailException, TextInputFilterAndReplace}
import is.hail.variant.{Locus, ReferenceGenome}
import is.hail.variant.BuiltinLoci.builtin

import scala.collection.immutable.ArraySeq

import org.apache.spark.sql.Row
import org.junit.jupiter.api.Test

class LoadVCFSuite {
  private def vcfReader(
    file: String,
    rg: Option[String] = Some(ReferenceGenome.GRCh37),
    skipInvalidLoci: Boolean = false,
  )(implicit ctx: ExecuteContext
  ): MatrixVCFReader =
    importVCF(getTestResource(file), rg = rg, skipInvalidLoci = skipInvalidLoci)
      .asInstanceOf[MatrixRead]
      .reader
      .asInstanceOf[MatrixVCFReader]

  @Test def testRowLocusTypeIsCanonical(implicit ctx: ExecuteContext): Unit = {
    val grch37 = builtin(ReferenceGenome.GRCh37)
    val results = ArraySeq.newBuilder[IndexedSeq[Row]]
    forEachCanonicalLocus { canonicalLocusType =>
      val (pt, rows) = readRowsLocally(vcfReader("sample.vcf"))
      assertEq(pt.fieldType("locus").setRequired(false), canonicalLocusType(grch37, false))
      assertEq(rows.length, 346)
      assertEq(Row.fromSeq(rows.head.toSeq.take(2)), Row(Locus("20", 10019093), FastSeq("A", "G")))
      assertEq(rows.last.getAs[Locus](0), Locus("20", 17970876))
      results += rows
    }
    val ArraySeq(struct, packed) = results.result(): @unchecked
    assertEq(packed, struct)
  }

  @Test def testWithoutReferenceGenomeLocusIsStruct(implicit ctx: ExecuteContext): Unit =
    forEachCanonicalLocus { _ =>
      val (pt, rows) = readRowsLocally(vcfReader("sample.vcf", rg = None), "locus")
      assertEq(
        pt.fieldType("locus"),
        PCanonicalStruct(true, "contig" -> PCanonicalString(true), "position" -> PInt32(true)),
      )
      assertEq(rows.head, Row(Row("20", 10019093)))
    }

  @Test def testSkipInvalidLoci(implicit ctx: ExecuteContext): Unit = {
    val grch37 = builtin(ReferenceGenome.GRCh37)
    forEachCanonicalLocus { canonicalLocusType =>
      val reader = vcfReader("skip_invalid_loci.vcf", skipInvalidLoci = true)
      val (pt, rows) = readRowsLocally(reader, "locus", "alleles")
      assertEq(pt.fieldType("locus").setRequired(false), canonicalLocusType(grch37, false))
      assertEq(rows, SkipInvalidLociResources.validRows)

      // The parser wraps the error in one that names the line.
      val e = intercept[HailException] {
        readRowsLocally(vcfReader("skip_invalid_loci.vcf"), "locus", "alleles")
      }
      val messages =
        Iterator.iterate[Throwable](e)(_.getCause).takeWhile(_ != null).map(_.getMessage)
      assert(messages.exists(_.contains("Invalid locus '0:1'")), e)
    }
  }

  @Test def testGVCFPartitionReaderLocusTypeIsCanonical(implicit ctx: ExecuteContext): Unit = {
    val grch38 = builtin(ReferenceGenome.GRCh38)
    val path = getTestResource("gvcfs/HG00096.g.vcf.gz")
    val header = LoadVCF.getVCFHeaderInfo(ctx.fs, path, null, null, null)
    forEachCanonicalLocus { canonicalLocusType =>
      val reader = GVCFPartitionReader(
        header,
        Set("PGT"),
        TFloat64,
        arrayElementsRequired = true,
        Some(grch38),
        Map.empty,
        skipInvalidLoci = false,
        TextInputFilterAndReplace(),
        MatrixType.entriesIdentifier,
        "__uid",
      )
      val context = MakeStruct(FastSeq(
        "fileNum" -> I32(0),
        "path" -> Str(path),
        "contig" -> Str("chr20"),
        "start" -> I32(17959479),
        "end" -> I32(17959640),
      ))
      val (pt, rows) =
        collectRowsLocally(ReadPartition(context, reader.fullRowType, reader))
      assertEq(pt.fieldType("locus").setRequired(false), canonicalLocusType(grch38, false))
      assertEq(
        rows.map(_.getAs[Locus](0)),
        FastSeq(17959479, 17959480, 17959640).map(Locus("chr20", _)),
      )
    }
  }
}
