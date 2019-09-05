package is.hail.expr.ir

import is.hail.ExecStrategy
import is.hail.HailSuite
import is.hail.TestUtils.assertEvalsTo
import is.hail.expr.types.virtual._
import is.hail.utils.{FastIndexedSeq, FastSeq}
import is.hail.variant.{Locus, ReferenceGenome}
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class LocusFunctionsSuite extends HailSuite {

  implicit val execStrats = ExecStrategy.javaOnly

  private def grch38: ReferenceGenome = ReferenceGenome.GRCh38
  private def tlocus = TLocus(grch38)
  private def tvariant = TStruct("locus" -> tlocus, "alleles" -> TArray(TString()))

  def locusIR: Apply = Apply("Locus", FastSeq(Str("chr22"), I32(1)), tlocus)

  def locus = Locus("chr22", 1, grch38)

  @Test def contig() {
    assertEvalsTo(invoke("contig", TString(), locusIR), locus.contig)
  }

  @Test def position() {
    assertEvalsTo(invoke("position", TInt32(), locusIR), locus.position)
  }

  @Test def isAutosomalOrPseudoAutosomal() {
    assertEvalsTo(invoke("isAutosomalOrPseudoAutosomal", TBoolean(), locusIR), locus.isAutosomalOrPseudoAutosomal(grch38))
  }

  @Test def isAutosomal() {
    assertEvalsTo(invoke("isAutosomal", TBoolean(), locusIR), locus.isAutosomal(grch38))
  }

  @Test def inYNonPar() {
    assertEvalsTo(invoke("inYNonPar", TBoolean(), locusIR), locus.inYNonPar(grch38))
  }

  @Test def inXPar() {
    assertEvalsTo(invoke("inXPar", TBoolean(), locusIR), locus.inXPar(grch38))
  }

  @Test def isMitochondrial() {
    assertEvalsTo(invoke("isMitochondrial", TBoolean(), locusIR), locus.isMitochondrial(grch38))
  }

  @Test def inXNonPar() {
    assertEvalsTo(invoke("inXNonPar", TBoolean(), locusIR), locus.inXNonPar(grch38))
  }

  @Test def inYPar() {
    assertEvalsTo(invoke("inYPar", TBoolean(), locusIR), locus.inYPar(grch38))
  }

  @Test def minRep() {
    val alleles = MakeArray(Seq(Str("AA"), Str("AT")), TArray(TString()))
    assertEvalsTo(invoke("min_rep", tvariant, locusIR, alleles), Row(Locus("chr22", 2), FastIndexedSeq("A", "T")))
    assertEvalsTo(invoke("min_rep", tvariant, locusIR, NA(TArray(TString()))), null)
  }
  
  @Test def globalPosition() {
    assertEvalsTo(invoke("locusToGlobalPos", TInt64(), locusIR), grch38.locusToGlobalPos(locus))
  }
  
  @Test def reverseGlobalPosition() {
    val globalPosition = 2824183054L
    assertEvalsTo(invoke("globalPosToLocus", tlocus, I64(globalPosition)), grch38.globalPosToLocus(globalPosition))
  }
}
