package is.hail.types.physical.stypes.concrete

import is.hail.TestUtils._
import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{EmitCodeBuilder, EmitFunctionBuilder}
import is.hail.scalacheck._
import is.hail.types.physical.{PCanonicalLocus, PLocus, PPackedLocus}
import is.hail.types.physical.LocusRepresentations._
import is.hail.types.physical.stypes.interfaces.SLocusValue

import org.junit.jupiter.api.Test
import org.scalacheck.Prop.forAll

class SPackedLocusSuite {

  def compile[T: TypeInfo](
    ctx: ExecuteContext,
    pt: PLocus,
  )(
    f: (EmitCodeBuilder, SLocusValue) => Value[T]
  ): (Region, Long) => T = {
    val fb = EmitFunctionBuilder[Region, Long, T](ctx, "locus_property")
    fb.emitWithBuilder(cb => f(cb, pt.loadCheapSCode(cb, fb.getCodeParam[Long](2)).asLocus))
    val g = fb.resultWithIndex()(ctx.theHailClassLoader, ctx.fs, ctx.taskContext, ctx.r)
    g(_, _)
  }

  @Test def testAccessorsAgreeWithReference(implicit ctx: ExecuteContext): Unit =
    for (rg <- references)
      withReference(rg) { ctx =>
        val pt = PPackedLocus(rg.name)
        val contigIdx = compile[Int](ctx, pt)((cb, l) => l.contigIdx(cb))
        val position = compile[Int](ctx, pt)((cb, l) => l.position(cb))
        val packed = compile[Long](ctx, pt)((cb, l) => l.packed(cb))
        val contig = compile[String](ctx, pt)((cb, l) => l.contig(cb).loadString(cb))
        val size = compile[Long](ctx, pt)((cb, l) => l.sizeToStoreInBytes(cb).value)

        check(forAll(genLocus(rg)) { l =>
          ctx.r.pool.scopedRegion { r =>
            val addr = pt.unstagedStoreJavaObject(ctx.stateManager, l, r)
            val idx = rg.getContigIndex(l.contig)
            assertEq(contigIdx(r, addr), idx)
            assertEq(position(r, addr), l.position)
            assertEq(packed(r, addr), (idx.toLong << 32) | l.position.toLong)
            assertEq(contig(r, addr), l.contig)
            assertEq(size(r, addr), pt.byteSize)
          }
          true
        })
      }

  @Test def testPrefixCodesMatchCanonical(implicit ctx: ExecuteContext): Unit =
    for (rg <- references)
      withReference(rg) { ctx =>
        def prefixCode(pt: PLocus) =
          compile[Array[Byte]](ctx, pt)((cb, l) =>
            l.prefixCode(cb).asInstanceOf[SJavaBytesValue].bytes
          )

        val canonical = PCanonicalLocus(rg.name)
        val packed = PPackedLocus(rg.name)
        val canonicalCode = prefixCode(canonical)
        val packedCode = prefixCode(packed)

        check(forAll(genLocus(rg)) { l =>
          ctx.r.pool.scopedRegion { r =>
            java.util.Arrays.equals(
              canonicalCode(r, canonical.unstagedStoreJavaObject(ctx.stateManager, l, r)),
              packedCode(r, packed.unstagedStoreJavaObject(ctx.stateManager, l, r)),
            )
          }
        })
      }

  @Test def testPrettyPrintsContigIndexAndPosition(implicit ctx: ExecuteContext): Unit =
    for (rg <- references)
      withReference(rg) { ctx =>
        val pt = PPackedLocus(rg.name)
        check(forAll(genLocus(rg)) { l =>
          ctx.r.pool.scopedRegion { r =>
            val addr = pt.unstagedStoreJavaObject(ctx.stateManager, l, r)
            Region.pretty(pt, addr) == s"#${rg.getContigIndex(l.contig)}:${l.position}"
          }
        })
      }
}
