package org.broadinstitute.hail.driver

import org.broadinstitute.hail.variant.GenotypeType._
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant.{GenotypeType, Genotype}
import org.kohsuke.args4j.{Option => Args4jOption}

object GQHistCombiner {
  def apply(): GQHistCombiner =
    new GQHistCombiner(Array.fill[Int](300)(0))

  def toIndex(gq: Int, gtType: GenotypeType): Int = {
    assert(gq >= 0 && gq < 100)
    val id = gtType.id
    assert(id >= 0 && id <= 3)

    gq + 100 * gtType.id
  }

  def fromIndex(i: Int): (Int, GenotypeType) = {
    val gq = i % 100
    val id = i / 100

    assert(gq >= 0 && gq < 100)
    assert(id >= 0 && id <= 3)

    (gq, GenotypeType(id))
  }
}

class GQHistCombiner(val a: Array[Int]) extends Serializable {

  import GQHistCombiner._

  def merge(g: Genotype): GQHistCombiner = {
    g.gq.foreach { gqx =>
      if (g.isCalled)
        a(toIndex(gqx, g.gtType)) += 1
    }

    this
  }

  def merge(comb: GQHistCombiner): GQHistCombiner = {
    for (i <- a.indices)
      a(i) += comb.a(i)

    this
  }
}

object GQHist extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _
  }

  def newOptions = new Options

  def name = "gqhist"

  def description = "Compute histogram of GQ values stratified over genotype call"

  def supportsMultiallelic = true

  def requiresVDS = true

  override def hidden = true

  def run(state: State, options: Options): State = {

    // FIXME move to tests
    for (gq <- 0 until 100)
      for (gtType <- Set(HomRef, Het, HomVar)) {
        assert((gq, gtType) == GQHistCombiner.fromIndex(GQHistCombiner.toIndex(gq, gtType)))
      }
    for (i <- 0 until 300) {
      val (gq, gtType) = GQHistCombiner.fromIndex(i)
      assert(i == GQHistCombiner.toIndex(gq, gtType))
    }

    val vds = state.vds

    val result = vds.rdd.map { case (v, va, gs) =>
      gs.aggregate(GQHistCombiner())((comb, g) =>
        comb.merge(g),
        (comb1, comb2) => comb1.merge(comb2))
    }.aggregate(GQHistCombiner())((comb1, comb2) => comb1.merge(comb2),
      (comb1, comb2) => comb1.merge(comb2))

    val a = result.a
    writeTextFile(options.output, state.hadoopConf) { s =>
      s.write("GQ\tGT\tCOUNT\n")
      for (i <- a.indices) {
        val (id, gtType) = GQHistCombiner.fromIndex(i)

        if (a(i) != 0)
          s.write(s"$id\t$gtType\t${a(i)}\n")
      }
    }

    state
  }
}
