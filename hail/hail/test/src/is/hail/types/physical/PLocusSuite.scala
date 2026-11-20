package is.hail.types.physical

import is.hail.TestUtils._
import is.hail.annotations.{Region, SafeRow}
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{EmitCodeBuilder, EmitFunctionBuilder}
import is.hail.scalacheck._
import is.hail.types.physical.LocusRepresentations._
import is.hail.types.physical.stypes.interfaces.SLocusValue
import is.hail.types.virtual.{TInterval, TLocus, TStruct}
import is.hail.variant.{Locus, ReferenceGenome}

import org.junit.jupiter.api.Test
import org.scalacheck.Prop.forAll

class PLocusSuite {

  /** Compiles `construct`, stores the locus it builds as `pt`, and reads it back. */
  private def compileConstruct[A: TypeInfo](
    ctx: ExecuteContext,
    pt: PLocus,
  )(
    construct: (EmitCodeBuilder, Value[Region], Value[A]) => SLocusValue
  ): (Region, A) => Locus = {
    val fb = EmitFunctionBuilder[Region, A, Long](ctx, "construct_locus")
    fb.emitWithBuilder { cb =>
      val r = fb.getCodeParam[Region](1)
      val l = construct(cb, r, fb.getCodeParam[A](2))
      assertEq(l.st, pt.sType)
      pt.store(cb, r, l, deepCopy = false)
    }
    val f = fb.resultWithIndex()(ctx.theHailClassLoader, ctx.fs, ctx.taskContext, ctx.r)
    (r, a) => SafeRow.read(pt, f(r, a)).asInstanceOf[Locus]
  }

  private def constructFromContigAndPosition(ctx: ExecuteContext, pt: PLocus)
    : (Region, Locus) => Locus =
    compileConstruct[Locus](ctx, pt) { (cb, r, l) =>
      pt.constructFromContigAndPosition(
        cb,
        r,
        l.invoke[String]("contig"),
        l.invoke[Int]("position"),
      )
    }

  private def locusTypes(rg: ReferenceGenome): IndexedSeq[PLocus] =
    representations(PCanonicalLocus(rg, required = true), rg).map(_.asInstanceOf[PLocus])

  @Test def testConstructFromContigAndPosition(implicit ctx: ExecuteContext): Unit =
    for {
      rg <- references
      pt <- locusTypes(rg)
    } {
      val f = constructFromContigAndPosition(ctx, pt)
      check(forAll(genLocus(rg))(l => ctx.r.pool.scopedRegion(r => f(r, l) == l)))
    }

  @Test def testConstructFromLocus(implicit ctx: ExecuteContext): Unit =
    for {
      rg <- references
      pt <- locusTypes(rg)
    } {
      val f = compileConstruct[Locus](ctx, pt)((cb, r, l) => pt.constructFromLocus(cb, r, l))
      check(forAll(genLocus(rg))(l => ctx.r.pool.scopedRegion(r => f(r, l) == l)))
    }

  // The constructor doesn't validate. A struct locus holds any contig; a packed locus can't.
  @Test def testConstructOnUnknownContig(implicit ctx: ExecuteContext): Unit =
    for (rg <- references) {
      val unknown = Locus("not_a_contig", 1)

      val struct = constructFromContigAndPosition(ctx, PCanonicalLocus(rg, required = true))
      ctx.r.pool.scopedRegion(r => assertEq(struct(r, unknown), unknown))

      val packed = constructFromContigAndPosition(ctx, PPackedLocus(rg, required = true))
      interceptFatal("Contig 'not_a_contig' is not in the reference genome") {
        ctx.r.pool.scopedRegion(r => packed(r, unknown))
      }
    }

  // Struct loci stay canonical until every locus producer can build a packed locus.
  @Test def testStructLocusIsCanonical(): Unit =
    for (rg <- references)
      assertEq(PLocus.canonical(rg, required = true), PCanonicalLocus(rg, required = true))

  @Test def testCanonicalPTypeOfLocusType(): Unit =
    forEachCanonicalLocus { canonicalLocusType =>
      for {
        rg <- references
        required <- Seq(false, true)
      } {
        assertEq(PType.canonical(TLocus(rg), required), canonicalLocusType(rg, required))
        assertEq(
          PType.canonical(TStruct("i" -> TInterval(TLocus(rg))), required, innerRequired = true),
          PCanonicalStruct(
            required,
            "i" -> PCanonicalInterval(canonicalLocusType(rg, true), required = true),
          ),
        )
      }
    }

  @Test def testCanonicalPTypeOfLocusPType(): Unit =
    forEachCanonicalLocus { canonicalLocusType =>
      for {
        rg <- references
        pt <- representations(PCanonicalLocus(rg, required = true), rg)
      } {
        assertEq(PType.canonical(pt), canonicalLocusType(rg, true))
        assertEq(
          PType.canonical(PCanonicalArray(pt)),
          PCanonicalArray(canonicalLocusType(rg, true)),
        )
      }
    }

  @Test def testLiteralPTypeOfLocus(): Unit =
    forEachCanonicalLocus { canonicalLocusType =>
      for (rg <- references) {
        val l = Locus(rg.contigs.head, 1)
        assertEq(PType.literalPType(TLocus(rg), l), canonicalLocusType(rg, true))
        assertEq(PType.literalPType(TLocus(rg), null), canonicalLocusType(rg, false))
      }
    }
}
