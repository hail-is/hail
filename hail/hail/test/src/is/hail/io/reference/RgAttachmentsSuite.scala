package is.hail.io.reference

import is.hail.TestUtils._
import is.hail.expr.ir.CompileCacheKey
import is.hail.expr.ir.defs.I32
import is.hail.io.fs.{FS, HadoopFS}
import is.hail.utils.SerializableHadoopConfiguration
import is.hail.variant.{ReferenceGenome, ReferenceGenomeEnv}

import scala.collection.immutable.ArraySeq

import org.apache.hadoop.conf.Configuration
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

class RgAttachmentsSuite {

  private val fs: FS = new HadoopFS(new SerializableHadoopConfiguration(new Configuration()))

  // The test chain maps contig 20 to chr12 and chr20.
  private def src: ReferenceGenome =
    ReferenceGenome("src", ArraySeq("20"), Map("20" -> 63025520))

  private def dest: ReferenceGenome =
    ReferenceGenome(
      "dest",
      ArraySeq("chr12", "chr20"),
      Map("chr12" -> 133275309, "chr20" -> 64444167),
    )

  // Both a source and a destination of the test chain.
  private def mid: ReferenceGenome =
    ReferenceGenome(
      "mid",
      ArraySeq("20", "chr12", "chr20"),
      Map("20" -> 63025520, "chr12" -> 133275309, "chr20" -> 64444167),
    )

  // The test FASTA has contigs a, b and c.
  private def fake: ReferenceGenome =
    ReferenceGenome("fake", ArraySeq("a", "b", "c"), Map("a" -> 25, "b" -> 15, "c" -> 10))

  private def sequence(rg: ReferenceGenome): ReferenceSequence =
    ReferenceSequence(
      fs,
      rg,
      getTestResource("fake_reference.fasta"),
      getTestResource("fake_reference.fasta.fai"),
    )

  private def liftover(source: ReferenceGenome, dest: ReferenceGenome): Liftover =
    Liftover(fs, source, dest, getTestResource("grch37_to_grch38_chr20.over.chain.gz"))

  @Test def testALiftoverIsFoundByEqualGenomes(): Unit = {
    val lo = liftover(src, dest)
    val (source, destination) = (src, dest)
    assertTrue(source ne lo.source)
    assertTrue(destination ne lo.dest)
    assertTrue(RgAttachments.empty.withLiftover(lo).liftover(source, destination) eq lo)
  }

  @Test def testASecondLiftoverForEqualGenomesFails(): Unit = {
    val attachments = RgAttachments.empty.withLiftover(liftover(src, dest))
    interceptFatal(
      "LiftOver already exists for source reference 'src' and destination reference 'dest'"
    ) {
      attachments.withLiftover(liftover(src, dest))
    }
  }

  @Test def testASequenceIsFoundByAnEqualGenome(): Unit = {
    val seq = sequence(fake)
    val rg = fake
    assertTrue(rg ne seq.rg)
    assertTrue(RgAttachments.empty.withSequence(seq).sequence(rg) eq seq)
  }

  @Test def testASecondSequenceForAnEqualGenomeFails(): Unit = {
    val attachments = RgAttachments.empty.withSequence(sequence(fake))
    interceptFatal("FASTA sequence has already been loaded for reference genome 'fake'") {
      attachments.withSequence(sequence(fake))
    }
  }

  @Test def testWithoutSequenceRemovesIt(): Unit = {
    val attachments = RgAttachments.empty.withSequence(sequence(fake)).withoutSequence(fake)
    assertEq(attachments, RgAttachments.empty)
  }

  @Test def testWithoutAnAbsentSequenceFails(): Unit =
    interceptFatal("Reference genome 'fake' does not have sequence loaded") {
      RgAttachments.empty.withoutSequence(fake)
    }

  @Test def testRemoveLiftoverRemovesIt(): Unit = {
    val env = ReferenceGenomeEnv(Seq(src, dest))
    val attachments = RgAttachments.empty.withLiftover(liftover(src, dest))
    assertEq(RgAttachments.removeLiftover(env, attachments, "src", "dest"), RgAttachments.empty)
  }

  @Test def testRemoveLiftoverWithAGenomeOutOfScopeIsANoOp(): Unit = {
    val attachments = RgAttachments.empty.withSequence(sequence(fake))
    val env = ReferenceGenomeEnv(Seq(src, fake))
    assertEq(RgAttachments.removeLiftover(env, attachments, "src", "dest"), attachments)
    assertEq(RgAttachments.removeLiftover(env, attachments, "dest", "src"), attachments)
  }

  @Test def testRemoveReferenceRemovesLiftoversFromAndToIt(): Unit = {
    val unrelated = liftover(src, dest)
    val seq = sequence(fake)
    val env = ReferenceGenomeEnv(Seq(src, mid, dest, fake))
    val attachments =
      RgAttachments.empty
        .withLiftover(liftover(src, mid))
        .withLiftover(liftover(mid, dest))
        .withLiftover(unrelated)
        .withSequence(seq)

    val (env2, attachments2) = RgAttachments.removeReference(env, attachments, "mid")
    assertEq(env2, ReferenceGenomeEnv(Seq(src, dest, fake)))
    assertEq(attachments2, RgAttachments.empty.withLiftover(unrelated).withSequence(seq))
  }

  @Test def testRemoveReferenceRemovesItsSequence(): Unit = {
    val unrelated = liftover(src, dest)
    val env = ReferenceGenomeEnv(Seq(src, dest, fake))
    val attachments = RgAttachments.empty.withLiftover(unrelated).withSequence(sequence(fake))

    val (_, attachments2) = RgAttachments.removeReference(env, attachments, "fake")
    assertEq(attachments2, RgAttachments.empty.withLiftover(unrelated))
  }

  @Test def testARedefinedGenomeHasNoLiftovers(): Unit = {
    val env = ReferenceGenomeEnv(Seq(src, mid, dest))
    val attachments =
      RgAttachments.empty.withLiftover(liftover(src, mid)).withLiftover(liftover(mid, dest))

    val (env2, attachments2) = RgAttachments.removeReference(env, attachments, "mid")
    val redefined = mid.copy(lengths = mid.lengths + ("20" -> 1))
    val env3 = env2.withReferences(Seq(redefined))

    assertTrue(env3.resolve("mid") eq redefined)
    assertTrue(!attachments2.liftovers.contains((src, redefined)))
    assertTrue(!attachments2.liftovers.contains((redefined, dest)))
    interceptFatal("No liftover from reference genome 'src' to 'mid'") {
      attachments2.liftover(src, redefined)
    }
  }

  @Test def testReaddingAnIdenticalGenomeResurrectsNothing(): Unit = {
    val env = ReferenceGenomeEnv(Seq(src, mid, dest, fake))
    val attachments =
      RgAttachments.empty
        .withLiftover(liftover(src, mid))
        .withLiftover(liftover(mid, dest))
        .withSequence(sequence(fake))

    val (env2, attachments2) = RgAttachments.removeReference(env, attachments, "mid")
    val (env3, attachments3) = RgAttachments.removeReference(env2, attachments2, "fake")
    val env4 = env3.withReferences(Seq(mid, fake))

    assertEq(env4, env)
    assertEq(attachments3, RgAttachments.empty)
  }

  // Compiled code holds the attachments it was compiled against, so a cache key with the same
  // attachment objects hits, and one with an equally defined but different liftover misses.
  @Test def testCompileCacheKeyComparesAttachmentsByIdentity(): Unit = {
    def key(attachments: RgAttachments): CompileCacheKey =
      CompileCacheKey(IndexedSeq.empty, Seq.empty, I32(0), attachments)

    val lo = liftover(src, dest)
    assertEq(key(RgAttachments.empty.withLiftover(lo)), key(RgAttachments.empty.withLiftover(lo)))
    assertTrue(
      key(RgAttachments.empty.withLiftover(lo)) !=
        key(RgAttachments.empty.withLiftover(liftover(src, dest)))
    )
  }
}
