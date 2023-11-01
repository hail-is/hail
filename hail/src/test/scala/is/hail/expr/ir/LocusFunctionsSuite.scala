package is.hail.expr.ir

import is.hail.{ExecStrategy, HailSuite}
import is.hail.types.virtual._
import is.hail.utils.{FastSeq, Interval}
import is.hail.variant.{Locus, ReferenceGenome}
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class LocusFunctionsSuite extends HailSuite {

  implicit val execStrats = ExecStrategy.javaOnly

  private def grch38: ReferenceGenome = ctx.getReference(ReferenceGenome.GRCh38)
  private def tlocus = TLocus(grch38.name)
  private def tvariant = TStruct("locus" -> tlocus, "alleles" -> TArray(TString))

  def locusIR: Apply = Apply("Locus", FastSeq(), FastSeq(Str("chr22"), I32(1)), tlocus, ErrorIDs.NO_ERROR)

  def locus = Locus("chr22", 1, grch38)

  @Test def testLocusTwoArgs(): Unit = {
    assertEvalsTo(invoke("Locus", TLocus(grch38.name), Str("chr22"), I32(1)), locus)
  }

  @Test def testLocusOneArg(): Unit = {
    assertEvalsTo(invoke("Locus", TLocus(grch38.name), Str("chr22:1")), locus)
  }

  @Test def contig() {
    assertEvalsTo(invoke("contig", TString, locusIR), locus.contig)
  }

  @Test def position() {
    assertEvalsTo(invoke("position", TInt32, locusIR), locus.position)
  }

  @Test def isAutosomalOrPseudoAutosomal() {
    assertEvalsTo(invoke("isAutosomalOrPseudoAutosomal", TBoolean, locusIR), locus.isAutosomalOrPseudoAutosomal(grch38))
  }

  @Test def isAutosomal() {
    assertEvalsTo(invoke("isAutosomal", TBoolean, locusIR), locus.isAutosomal(grch38))
  }

  @Test def inYNonPar() {
    assertEvalsTo(invoke("inYNonPar", TBoolean, locusIR), locus.inYNonPar(grch38))
  }

  @Test def inXPar() {
    assertEvalsTo(invoke("inXPar", TBoolean, locusIR), locus.inXPar(grch38))
  }

  @Test def isMitochondrial() {
    assertEvalsTo(invoke("isMitochondrial", TBoolean, locusIR), locus.isMitochondrial(grch38))
  }

  @Test def inXNonPar() {
    assertEvalsTo(invoke("inXNonPar", TBoolean, locusIR), locus.inXNonPar(grch38))
  }

  @Test def inYPar() {
    assertEvalsTo(invoke("inYPar", TBoolean, locusIR), locus.inYPar(grch38))
  }

  @Test def minRep() {
    val alleles = MakeArray(FastSeq(Str("AA"), Str("AT")), TArray(TString))
    assertEvalsTo(invoke("min_rep", tvariant, locusIR, alleles), Row(Locus("chr22", 2), FastSeq("A", "T")))
    assertEvalsTo(invoke("min_rep", tvariant, locusIR, NA(TArray(TString))), null)
  }

  @Test def globalPosition() {
    assertEvalsTo(invoke("locusToGlobalPos", TInt64, locusIR), grch38.locusToGlobalPos(locus))
  }

  @Test def reverseGlobalPosition() {
    val globalPosition = 2824183054L
    assertEvalsTo(invoke("globalPosToLocus", tlocus, I64(globalPosition)), grch38.globalPosToLocus(globalPosition))
  }

  @Test def testMultipleReferenceGenomes() {
    implicit val execStrats = ExecStrategy.compileOnly

    val ir = MakeTuple.ordered(FastSeq(
      invoke("Locus", TLocus(ReferenceGenome.GRCh37), Str("1"), I32(1)),
      invoke("Locus", TLocus(ReferenceGenome.GRCh38), Str("chr1"), I32(1))))

    assertEvalsTo(ir, Row(Locus("1", 1, ctx.getReference(ReferenceGenome.GRCh37)), Locus("chr1", 1, ctx.getReference(ReferenceGenome.GRCh38))))
  }

  @Test def testMakeInterval() {
    // TString, TInt32, TInt32, TBoolean, TBoolean, TBoolean
    val ir = MakeTuple.ordered(FastSeq(
      invoke("LocusInterval", TInterval(TLocus(grch38.name)), NA(TString), I32(1), I32(100), True(), True(), False()),
      invoke("LocusInterval", TInterval(TLocus(grch38.name)), Str("chr1"), NA(TInt32), I32(100), True(), True(), False()),
      invoke("LocusInterval", TInterval(TLocus(grch38.name)), Str("chr1"), I32(1), NA(TInt32), True(), True(), False()),
      invoke("LocusInterval", TInterval(TLocus(grch38.name)), Str("chr1"), I32(1), I32(100), NA(TBoolean), True(), False()),
      invoke("LocusInterval", TInterval(TLocus(grch38.name)), Str("chr1"), I32(1), I32(100), True(), NA(TBoolean), False()),
      invoke("LocusInterval", TInterval(TLocus(grch38.name)), Str("chr1"), I32(1), I32(100), True(), True(), NA(TBoolean)),
      invoke("LocusInterval", TInterval(TLocus(grch38.name)), Str("chr1"), I32(-1), I32(0), True(), True(), True()),
      invoke("LocusInterval", TInterval(TLocus(grch38.name)), Str("chr1"), I32(1), I32(100), True(), True(), True())
    ))

    assertEvalsTo(ir,
      Row(
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        Interval(Locus("chr1", 1, grch38), Locus("chr1", 100, grch38), true, true)
      )
    )
  }
}
