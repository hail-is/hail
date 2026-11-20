package is.hail.io.bgen

import is.hail.TestUtils._
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.io.SkipInvalidLociResources
import is.hail.io.bgen.BgenIndexSuite.indexBgen
import is.hail.types.physical.LocusRepresentations.forEachCanonicalLocus
import is.hail.types.virtual.TStruct
import is.hail.variant.{Locus, ReferenceGenome}
import is.hail.variant.BuiltinLoci.builtin

import scala.collection.immutable.ArraySeq

import org.apache.spark.sql.Row
import org.junit.jupiter.api.Test

class LoadBgenSuite {
  private def grch37(implicit ctx: ExecuteContext): ReferenceGenome =
    builtin(ReferenceGenome.GRCh37)

  private def reader(name: String)(implicit ctx: ExecuteContext): MatrixBGENReader = {
    val bgen = getTestResource(name)
    MatrixBGENReader(
      ctx,
      FastSeq(bgen),
      None,
      Map(bgen -> getTestResource(s"$name.idx2")),
      Some(3),
      None,
      None,
    )
  }

  @Test def testRowLocusTypeIsCanonical(implicit ctx: ExecuteContext): Unit = {
    val results = ArraySeq.newBuilder[IndexedSeq[Row]]
    forEachCanonicalLocus { canonicalLocusType =>
      val (pt, rows) = readRowsLocally(reader("example.8bits.bgen"))
      assertEq(pt.fieldType("locus").setRequired(false), canonicalLocusType(grch37, false))
      assertEq(rows.length, 199)
      assertEq(
        Row.fromSeq(rows.head.toSeq.take(4)),
        Row(Locus("1", 1001), FastSeq("A", "G"), "RSID_101", "SNPID_101"),
      )
      assertEq(rows.last.getAs[Locus](0), Locus("1", 100001))
      results += rows
    }
    val ArraySeq(struct, packed) = results.result(): @unchecked
    assertEq(packed, struct)
  }

  // The row PType in the settings decides requiredness and the key type the index writer is built
  // with; neither shows in the rows read or in the bytes of an index.
  @Test def testSettingsLocusTypeIsCanonical(implicit ctx: ExecuteContext): Unit =
    forEachCanonicalLocus { canonicalLocusType =>
      val requestedType =
        MatrixBGENReader.fullMatrixTypeWithoutUIDs(Some(grch37)).canonicalTableType
      val settings = BgenSettings(0, requestedType, Some(grch37), TStruct.empty)
      assertEq(settings.rowPType.fieldType("locus"), canonicalLocusType(grch37, false))
      assertEq(settings.indexKeyType.fieldType("locus"), canonicalLocusType(grch37, false))
    }

  @Test def testSkipInvalidLoci(implicit ctx: ExecuteContext): Unit =
    forEachCanonicalLocus { canonicalLocusType =>
      // The genotypes have 16-bit probabilities, which the reader can't decode.
      val (pt, rows) = readRowsLocally(reader("skip_invalid_loci.bgen"), "locus", "alleles")
      assertEq(pt.fieldType("locus").setRequired(false), canonicalLocusType(grch37, false))
      assertEq(rows, SkipInvalidLociResources.validRows)
    }

  @Test def testInvalidLociFailIndexing(implicit ctx: ExecuteContext): Unit =
    forEachCanonicalLocus { _ =>
      val idx = ctx.createTmpPath("test-bgen-invalid-loci", "idx2")
      interceptFatal("Invalid locus '0:1'") {
        indexBgen(getTestResource("skip_invalid_loci.bgen"), idx, Map.empty)
      }
    }
}
