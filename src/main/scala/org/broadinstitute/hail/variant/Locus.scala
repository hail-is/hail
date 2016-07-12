package org.broadinstitute.hail.variant

import org.broadinstitute.hail.check.Gen

object Locus {
  val simpleContigs: Seq[String] = (1 to 22).map(_.toString) ++ Seq("X", "Y", "MT")

  def gen(contigs: Seq[String]): Gen[Locus] =
    Gen.zip(Gen.oneOfSeq(contigs), Gen.posInt)
      .map { case (contig, pos) => Locus(contig, pos) }

  def gen: Gen[Locus] = gen(simpleContigs)
}

case class Locus(contig: String, position: Int) extends Ordered[Locus] {
  def compare(that: Locus): Int = {
    var c = Contig.compare(contig, that.contig)
    if (c != 0)
      return c

    position.compare(that.position)
  }
}
