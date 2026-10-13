package is.hail.variant

import is.hail.TestUtils._
import is.hail.annotations.Region
import is.hail.backend.ExecuteContext

import scala.collection.immutable.ArraySeq

import org.junit.jupiter.api.Test
import org.scalacheck.Gen
import org.scalacheck.Prop.forAll

class ContigIndexCacheSuite {

  // Contigs that share prefixes and lengths, so that a comparison that stops early or checks only
  // the length gives a wrong index.
  private val custom: ReferenceGenome = {
    val contigs = ArraySeq("1", "10", "11", "chr1", "chr10", "chr2", "a", "ab", "ba")
    ReferenceGenome("contig_index_cache_test", contigs, contigs.map(_ -> 1000).toMap)
  }

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
}
