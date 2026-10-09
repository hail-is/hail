package is.hail.types.physical

import is.hail.ParameterizedTest
import is.hail.TestUtils._
import is.hail.annotations.{Region, RowSeq, SafeRow, UnsafeRow}
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir.{EmitFunctionBuilder, IRParser}
import is.hail.rvd.AbstractRVDSpec
import is.hail.types.physical.LocusRepresentations._
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant.ReferenceGenome

import scala.collection.immutable.ArraySeq

import org.json4s.jackson.Serialization
import org.junit.jupiter.api.Test
import org.scalacheck.Prop.forAll

class PTypeSuite {

  def ptypes() = ArraySeq[PType](
    PInt32(true),
    PInt32(false),
    PInt64(true),
    PInt64(false),
    PFloat32(true),
    PFloat64(true),
    PBoolean(true),
    PCanonicalCall(true),
    PCanonicalBinary(false),
    PCanonicalString(true),
    PCanonicalLocus(ReferenceGenome.GRCh37, false),
    PCanonicalArray(PInt32Required, true),
    PCanonicalSet(PInt32Required, false),
    PCanonicalDict(PInt32Required, PCanonicalString(true), true),
    PCanonicalInterval(PInt32Optional, false),
    PCanonicalTuple(
      FastSeq(PTupleField(1, PInt32Required), PTupleField(3, PCanonicalString(false))),
      true,
    ),
    PCanonicalStruct(
      FastSeq(PField("foo", PInt32Required, 0), PField("bar", PCanonicalString(false), 1)),
      true,
    ),
  )

  @Test def testPTypesDataProvider(): Unit = ptypes(): Unit

  @ParameterizedTest("ptypes")
  def testSerialization(ptype: PType): Unit = {
    implicit val formats = AbstractRVDSpec.formats
    val s = Serialization.write(ptype)
    assertEq(Serialization.read[PType](s), ptype)
  }

  @Test def testLiteralPType(): Unit = {
    assertEq(PType.literalPType(TInt32, 5), PInt32(true))
    assertEq(PType.literalPType(TInt32, null), PInt32())

    assertEq(PType.literalPType(TArray(TInt32), null), PCanonicalArray(PInt32(true)))
    assertEq(PType.literalPType(TArray(TInt32), FastSeq(1, null)), PCanonicalArray(PInt32(), true))
    assertEq(PType.literalPType(TArray(TInt32), FastSeq(1, 5)), PCanonicalArray(PInt32(true), true))

    assertEq(
      PType.literalPType(
        TInterval(TInt32),
        Interval(5, null, false, true),
      ),
      PCanonicalInterval(PInt32(), true),
    )

    val p = TStruct("a" -> TInt32, "b" -> TInt32)
    val d = TDict(p, p)
    assertEq(
      PType.literalPType(d, Map(RowSeq(3, null) -> RowSeq(null, 3))),
      PCanonicalDict(
        PCanonicalStruct(true, "a" -> PInt32(true), "b" -> PInt32()),
        PCanonicalStruct(true, "a" -> PInt32(), "b" -> PInt32(true)),
        true,
      ),
    )
  }

  @Test def testPackedLocusParsesAsIRText(): Unit =
    for {
      rg <- ArraySeq(ReferenceGenome.GRCh38, customReference.name)
      pt <- ArraySeq[PType](
        PPackedLocus(rg),
        PPackedLocus(rg, required = true),
        PCanonicalStruct("l" -> PPackedLocus(rg)),
        PCanonicalInterval(PPackedLocus(rg, required = true)),
      )
    } assertEq(IRParser.parsePType(pt.toString), pt)

  def compileStore(ctx: ExecuteContext, src: PType, dst: PType, deepCopy: Boolean, coerce: Boolean)
    : (Region, Long) => Long = {
    val fb = EmitFunctionBuilder[Region, Long, Long](ctx, "locus_representation_store")
    fb.emitWithBuilder { cb =>
      val r = fb.getCodeParam[Region](1)
      val v = src.loadCheapSCode(cb, fb.getCodeParam[Long](2))
      if (coerce)
        dst.store(cb, r, dst.sType.coerceOrCopy(cb, r, v, deepCopy), deepCopy = false)
      else
        dst.store(cb, r, v, deepCopy)
    }
    val f = fb.resultWithIndex()(ctx.theHailClassLoader, ctx.fs, ctx.taskContext, ctx.r)
    f(_, _)
  }

  // Loci in either representation, nested anywhere, survive store, copy, deep copy and coercion
  // into either representation, and read back as the original value.
  @Test def testLocusRepresentationsRoundTrip(implicit ctx: ExecuteContext): Unit =
    for (rg <- references)
      withReference(rg) { ctx =>
        val sm = ctx.stateManager
        check(forAll(genPTypeValOn(ctx, rg)) { case (pt, a) =>
          val representations = ArraySeq(toCanonical(pt), toPacked(pt))
          for {
            src <- representations
            dst <- representations
            deepCopy <- ArraySeq(false, true)
          } ctx.r.pool.scopedRegion { r =>
            def assertReadsBack(addr: Long, how: String): Unit = {
              val clue = s"$how: $src -> $dst, deepCopy=$deepCopy"
              val t = dst.virtualType
              assert(t.valuesSimilar(SafeRow.read(sm, dst, addr), a), clue)
              assert(t.valuesSimilar(UnsafeRow.read(sm, dst, r, addr), a), clue)
            }

            val srcAddr = src.unstagedStoreJavaObject(sm, a, r)
            assertReadsBack(dst.copyFromAddress(sm, r, src, srcAddr, deepCopy), "unstaged copy")
            assertReadsBack(
              compileStore(ctx, src, dst, deepCopy, coerce = false)(r, srcAddr),
              "staged store",
            )
            assertReadsBack(
              compileStore(ctx, src, dst, deepCopy, coerce = true)(r, srcAddr),
              "staged coerce",
            )
          }
          true
        })
      }
}
