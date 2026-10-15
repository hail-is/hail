package is.hail.variant

import is.hail.ExecStrategy
import is.hail.TestUtils._
import is.hail.annotations.{Region, RowSeq}
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir._
import is.hail.expr.ir.TestUtils.collect
import is.hail.expr.ir.defs.{Literal, TableWrite}
import is.hail.types.physical.LocusRepresentations.withReference
import is.hail.types.physical.PCanonicalLocus
import is.hail.types.physical.stypes.interfaces.SLocusValue
import is.hail.types.virtual.{TArray, TInt32, TLocus, TString}
import is.hail.variant.ContigIndexCacheSuite.{
  compileCanonical, custom, locusFromLiterals, storeCanonical,
}

import scala.collection.immutable.ArraySeq

import org.junit.jupiter.api.Test
import org.scalacheck.Gen
import org.scalacheck.Prop.forAll

class ContigIndexCacheSuite {

  private def genLookups(rg: ReferenceGenome): Gen[List[String]] =
    Gen.listOf(Gen.zip(Gen.oneOf(rg.contigs), Gen.choose(1, 4))).map(_.flatMap {
      case (contig, run) => List.fill(run)(contig)
    })

  /* Every lookup writes its contig to the same address, so a cache that kept the address instead of
   * a copy of the bytes would see its entry change underneath it. */
  private def checkAgreesWithReferenceGenome(ctx: ExecuteContext, rg: ReferenceGenome): Unit = {
    val maxLength = rg.contigs.map(_.getBytes.length).max
    val buf = ctx.r.allocate(maxLength.toLong)
    check(forAll(genLookups(rg)) { contigs =>
      val cache = new ContigIndexCache(rg)
      contigs.foreach { contig =>
        val bytes = contig.getBytes
        Region.storeBytes(buf, bytes)
        assertEq(cache.contigIndex(buf, bytes.length), rg.getContigIndex(contig))
      }
      true
    })
  }

  @Test def testAgreesWithBuiltinReferenceGenome(implicit ctx: ExecuteContext): Unit =
    checkAgreesWithReferenceGenome(ctx, ctx.references(ReferenceGenome.GRCh38))

  @Test def testAgreesWithCustomReferenceGenome(implicit ctx: ExecuteContext): Unit =
    checkAgreesWithReferenceGenome(ctx, custom)

  private def outcome(f: => Int): Either[(Class[_], String), Int] =
    try Right(f)
    catch { case e: Throwable => Left((e.getClass, e.getMessage)) }

  @Test def testUnknownContigBehavesLikeReferenceGenome(implicit ctx: ExecuteContext): Unit = {
    val cache = new ContigIndexCache(custom)
    val buf = ctx.r.allocate(8L)
    def lookup(contig: String): Int = {
      val bytes = contig.getBytes
      Region.storeBytes(buf, bytes)
      cache.contigIndex(buf, bytes.length)
    }

    assertEq(lookup("chr1"), 3)
    // "chr" is a prefix of the cached contig, "chr3" has its length.
    for (unknown <- Seq("chr", "chr3", "chr3"))
      assertEq(outcome(lookup(unknown)), outcome(custom.getContigIndex(unknown)))
    // An unknown contig leaves the cache usable.
    assertEq(lookup("chr1"), 3)
    assertEq(lookup("chr10"), 4)
  }

  /* Clearing the region and storing a contig of the same length at the same address would fool a
   * cache keyed on the address, or on a view of the region, instead of a copy of the bytes. */
  @Test def testRegionReuse(implicit ctx: ExecuteContext): Unit = {
    val rg = ctx.references(ReferenceGenome.GRCh38)
    val pt = PCanonicalLocus(rg)
    val f = compileCanonical(ctx, rg)((cb, l) => cb.memoize(l.contigIdx(cb).toL))
    ctx.r.pool.scopedRegion { r =>
      val first = storeCanonical(ctx, rg, Locus("chr1", 1), r)
      val firstContig = pt.representation.loadField(first, 0)
      assertEq(f(first), rg.getContigIndex("chr1").toLong)

      r.clear()
      val second = storeCanonical(ctx, rg, Locus("chr2", 1), r)
      assertEq(second, first)
      assertEq(pt.representation.loadField(second, 0), firstContig)
      assertEq(f(second), rg.getContigIndex("chr2").toLong)
    }
  }

  @Test def testUnknownContigInGeneratedCode(implicit ctx: ExecuteContext): Unit = {
    val rg = ctx.references(ReferenceGenome.GRCh38)
    val f = compileCanonical(ctx, rg)((cb, l) => cb.memoize(l.contigIdx(cb).toL))
    def lookup(contig: String): Long = f(storeCanonical(ctx, rg, Locus(contig, 1), ctx.r))

    assertEq(lookup("chr1"), 0L)
    // "chrZ" has the cached contig's length, "chr" is a prefix of it.
    for (unknown <- Seq("chrZ", "chrZ", "chr"))
      assertEq(outcome(lookup(unknown).toInt), outcome(rg.getContigIndex(unknown)))
    assertEq(lookup("chr1"), 0L)
    assertEq(lookup("chr10"), 9L)
  }

  /* Partitions run concurrently on the local backend, each with its own instance of the generated
   * code. Contigs change every two rows, so lookups both hit and miss. Under unstable encodings,
   * writing packs each canonical locus through the cache, so a wrong index reads back as a wrong
   * contig. */
  @Test def testConcurrentPartitions(implicit ctx: ExecuteContext): Unit = {
    implicit val execStrats = ExecStrategy.lowering
    withReference(custom) { implicit ctx =>
      withUnstableEncodings { implicit ctx =>
        val n = 64 * 40
        val contigs = ArraySeq.tabulate(n)(i => custom.contigs((i / 2) % custom.contigs.length))
        val positions = ArraySeq.tabulate(n)(i => i % 997 + 1)
        val table = TableRange(n, 64).mapRows { (_, row) =>
          row.get("idx").bind { idx =>
            locusFromLiterals(contigs, positions)(idx).bind { locus =>
              makestruct(
                "idx" -> idx,
                "locus" -> locus,
                "contig_idx" -> invoke("contig_idx", TInt32, locus),
              )
            }
          }
        }

        val path = ctx.createTmpPath("contig-index-cache-concurrency", "ht")
        assertEvalsTo(TableWrite(table, TableNativeWriter(path)), ())
        val expected = ArraySeq.tabulate(n) { i =>
          RowSeq(i, Locus(contigs(i), positions(i)), custom.getContigIndex(contigs(i)))
        }
        val read = TableIR.read(ctx.fs, ctx.referenceGenomes, path, requestedType = Some(table.typ))
        assertEvalsTo(collect(read), RowSeq(expected, RowSeq()))
      }
    }
  }
}

object ContigIndexCacheSuite {

  // Contigs that share prefixes and lengths, so that a comparison that stops early or checks only
  // the length gives a wrong index.
  val custom: ReferenceGenome = {
    val contigs = ArraySeq("1", "10", "11", "chr1", "chr10", "chr2", "a", "ab", "ba")
    ReferenceGenome("contig_index_cache_test", contigs, contigs.map(_ -> 1000).toMap)
  }

  /** Constructs a canonical locus on [[custom]] from the `idx`th contig and position. */
  def locusFromLiterals(contigs: IndexedSeq[String], positions: IndexedSeq[Int])(idx: IR): IR =
    invoke(
      "Locus",
      TLocus(custom),
      Literal(TArray(TString), contigs).at(idx),
      Literal(TArray(TInt32), positions).at(idx),
    )

  type LocusEmitter = (EmitCodeBuilder, SLocusValue) => Value[Long]

  def storeCanonical(ctx: ExecuteContext, rg: ReferenceGenome, l: Locus, r: Region): Long =
    PCanonicalLocus(rg).unstagedStoreJavaObject(ctx.stateManager, l, r)

  /** Compiles `f` over a canonical locus, given its address. Each call of the result runs the same
    * generated instance, so a cache it holds persists across calls.
    */
  def compileCanonical(
    ctx: ExecuteContext,
    rg: ReferenceGenome,
  )(
    f: LocusEmitter
  ): AsmFunction1[Long, Long] = {
    val pt = PCanonicalLocus(rg)
    val fb = EmitFunctionBuilder[Long, Long](ctx, "canonical_locus")
    fb.emitWithBuilder(cb => f(cb, pt.loadCheapSCode(cb, fb.getCodeParam[Long](1)).asLocus))
    fb.resultWithIndex()(ctx.theHailClassLoader, ctx.fs, ctx.taskContext, ctx.r)
  }
}
