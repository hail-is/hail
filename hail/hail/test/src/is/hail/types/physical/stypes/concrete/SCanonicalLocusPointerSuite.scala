package is.hail.types.physical.stypes.concrete

import is.hail.TestUtils._
import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{EmitCodeBuilder, EmitFunctionBuilder}
import is.hail.scalacheck._
import is.hail.types.physical.PCanonicalLocus
import is.hail.types.physical.stypes.interfaces.SLocusValue
import is.hail.variant.{Locus, ReferenceGenome}

import scala.collection.immutable.ArraySeq

import org.junit.jupiter.api.Test
import org.scalacheck.Arbitrary.arbitrary
import org.scalacheck.Gen
import org.scalacheck.Prop.forAll

class SCanonicalLocusPointerSuite {

  def compile[T: TypeInfo](
    ctx: ExecuteContext,
    pt: PCanonicalLocus,
  )(
    f: (EmitCodeBuilder, SLocusValue) => Value[T]
  ): (Region, Long) => T = {
    val fb = EmitFunctionBuilder[Region, Long, T](ctx, "locus_property")
    fb.emitWithBuilder(cb => f(cb, pt.loadCheapSCode(cb, fb.getCodeParam[Long](2))))
    val g = fb.resultWithIndex()(ctx.theHailClassLoader, ctx.fs, ctx.taskContext, ctx.r)
    g(_, _)
  }

  def checkAgreesWithReference(rg: ReferenceGenome)(implicit ctx: ExecuteContext): Unit =
    ctx.local(references = ctx.references + (rg.name -> rg)) { ctx =>
      val pt = PCanonicalLocus(rg.name)
      val contigIdx = compile[Int](ctx, pt)((cb, l) => l.contigIdx(cb))
      val packed = compile[Long](ctx, pt)((cb, l) => l.packed(cb))

      check(forAll(genLocus(rg)) { l =>
        ctx.r.pool.scopedRegion { r =>
          val addr = pt.unstagedStoreJavaObject(ctx.stateManager, l, r)
          val idx = rg.getContigIndex(l.contig)
          assertEq(contigIdx(r, addr), idx)
          assertEq(packed(r, addr), (idx.toLong << 32) | l.position.toLong)
        }
      })
    }

  @Test def testContigIdxAndPackedGRCh37(implicit ctx: ExecuteContext): Unit =
    checkAgreesWithReference(ctx.references(ReferenceGenome.GRCh37))

  @Test def testContigIdxAndPackedGRCh38(implicit ctx: ExecuteContext): Unit =
    checkAgreesWithReference(ctx.references(ReferenceGenome.GRCh38))

  @Test def testContigIdxAndPackedCustomReference(implicit ctx: ExecuteContext): Unit =
    checkAgreesWithReference(
      ReferenceGenome("custom", ArraySeq("c", "a", "b"), Map("a" -> 100, "b" -> 2000, "c" -> 30000))
    )

  val referenceAndLoci: Gen[(ReferenceGenome, IndexedSeq[Locus])] =
    for {
      rg <- arbitrary[ReferenceGenome]
      loci <- Gen.containerOf[IndexedSeq, Locus](genLocus(rg))
    } yield (rg, loci)

  @Test def testPrefixCodePreservesLocusOrder(implicit ctx: ExecuteContext): Unit =
    check(forAll(referenceAndLoci) { case (rg, loci) =>
      ctx.local(references = ctx.references + (rg.name -> rg)) { ctx =>
        val pt = PCanonicalLocus(rg.name)
        val prefixCode =
          compile[Array[Byte]](ctx, pt)((cb, l) =>
            l.prefixCode(cb).asInstanceOf[SJavaBytesValue].bytes
          )

        val byPrefixCode = ctx.r.pool.scopedRegion { r =>
          loci
            .map(l => (l, prefixCode(r, pt.unstagedStoreJavaObject(ctx.stateManager, l, r))))
            .sortWith { case ((_, b1), (_, b2)) => java.util.Arrays.compareUnsigned(b1, b2) < 0 }
            .map(_._1)
        }

        assertEq(byPrefixCode, loci.sorted(rg.locusOrdering))
      }
    })
}
