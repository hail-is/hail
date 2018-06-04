package is.hail.expr.ir

import is.hail.TestUtils.assertEvalsTo
import is.hail.utils.FastSeq
import is.hail.variant.{Locus, ReferenceGenome}
import org.testng.annotations.Test
import org.scalatest.testng.TestNGSuite

class LocusFunctionsSuite extends TestNGSuite {

  val grch38: ReferenceGenome = ReferenceGenome.GRCh38

  val locusIR: Apply = {
    val fn = grch38.wrapFunctionName("Locus")
    Apply(fn, FastSeq(Str("chr22"), I32(1)))
  }

  val locus = Locus("chr22", 1, grch38)

  @Test def contig() {
    assertEvalsTo(invoke("contig", locusIR), locus.contig)
  }

  @Test def position() {
    assertEvalsTo(invoke("position", locusIR), locus.position)
  }

  @Test def isAutosomalOrPseudoAutosomal() {
    assertEvalsTo(invoke("isAutosomalOrPseudoAutosomal", locusIR), locus.isAutosomalOrPseudoAutosomal(grch38))
  }

  @Test def isAutosomal() {
    assertEvalsTo(invoke("isAutosomal", locusIR), locus.isAutosomal(grch38))
  }

  @Test def inYNonPar() {
    assertEvalsTo(invoke("inYNonPar", locusIR), locus.inYNonPar(grch38))
  }

  @Test def inXPar() {
    assertEvalsTo(invoke("inXPar", locusIR), locus.inXPar(grch38))
  }

  @Test def isMitochondrial() {
    assertEvalsTo(invoke("isMitochondrial", locusIR), locus.isMitochondrial(grch38))
  }

  @Test def inXNonPar() {
    assertEvalsTo(invoke("inXNonPar", locusIR), locus.inXNonPar(grch38))
  }

  @Test def inYPar() {
    assertEvalsTo(invoke("inYPar", locusIR), locus.inYPar(grch38))
  }
}
