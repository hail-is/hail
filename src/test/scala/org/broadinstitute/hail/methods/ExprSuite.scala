package org.broadinstitute.hail.methods

import org.broadinstitute.hail.{FatalException, SparkSuite}
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant.Genotype
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

import scala.collection.mutable.ArrayBuffer

class ExprSuite extends SparkSuite {

  @Test def exprTest() {
    val symTab = Map("i" ->(0, TInt),
      "j" ->(1, TInt),
      "d" ->(2, TDouble),
      "d2" ->(3, TDouble),
      "s" ->(4, TString),
      "s2" ->(5, TString),
      "a" ->(6, TArray(TInt)),
      "m" ->(7, TInt),
      "as" ->(8, TArray(TStruct(("a", TInt),
        ("b", TString)))),
      "gs" ->(9, TStruct(("noCall", TGenotype),
        ("homRef", TGenotype),
        ("het", TGenotype),
        ("homVar", TGenotype),
        ("hetNonRef35", TGenotype))),
      "t" ->(10, TBoolean),
      "f" ->(11, TBoolean),
      "mb" ->(12, TBoolean))
    val ec = EvalContext(symTab)

    val a = ec.a
    a(0) = 5 // i
    a(1) = -7 // j
    a(2) =  3.14
    a(3) =  5.79e7
    a(4) =  "12,34,56,78"
    a(5) =  "this is a String, there are many like it, but this one is mine"
    a(6) =  IndexedSeq(1, 2, null, 6, 3, 3, -1, 8)
    a(7) =  null // m
    a(8) =  (Array[Any](Annotation(23, "foo"),
      Annotation(-7, null)): IndexedSeq[Any])
    a(9) = Annotation(
      Genotype(),
      Genotype(gt = Some(0)),
      Genotype(gt = Some(1)),
      Genotype(gt = Some(2)),
      Genotype(gt = Some(Genotype.gtIndex(3, 5))))
    a(10) = true
    a(11) =false
    a(12) = null // mb
    assert(a.length == 13)

    def eval[T](s: String): T = {
      val f = Parser.parse[T](ec, null, s)
      f()
    }

    assert(eval[Boolean]("(1 / 2) == 0.5"))
    assert(eval[Boolean]("(1.0 / 2.0) == 0.5"))
    assert(eval[Boolean]("(1 / 2.0) == 0.5"))
    assert(eval[Boolean]("(1.0 / 2) == 0.5"))
    assert(eval[Boolean]("gs.noCall.gt.isMissing"))
    assert(eval[Boolean]("gs.noCall.gtj.isMissing"))
    assert(eval[Boolean]("gs.noCall.gtk.isMissing"))

    assert(eval[Int]("let a = i and b = j in a + b") == -2)
    assert(eval[Int]("let a = i and b = a + j in b") == -2)
    assert(eval[Int]("let i = j in i") == -7)
    assert(eval[Int]("let a = let b = j in b + 1 in a + 1") == -5)

    assert(eval[Boolean]("mb || true"))
    assert(eval[Boolean]("true || mb"))
    assert(eval[Boolean]("(false || mb).isMissing"))
    assert(eval[Boolean]("(mb || false).isMissing"))

    assert(eval[Int]("gs.homRef.gtj") == 0
      && eval[Int]("gs.homRef.gtk") == 0)
    assert(eval[Int]("gs.het.gtj") == 0
      && eval[Int]("gs.het.gtk") == 1)
    assert(eval[Int]("gs.homVar.gtj") == 1
      && eval[Int]("gs.homVar.gtk") == 1)
    assert(eval[Int]("gs.hetNonRef35.gtj") == 3
      && eval[Int]("gs.hetNonRef35.gtk") == 5)

    assert(eval[Int]("i.orElse(3)") == 5)
    assert(eval[Int]("m.orElse(3)") == 3)

    assert(!eval[Boolean]("i.isMissing"))
    assert(eval[Boolean]("i.isNotMissing"))

    assert(eval[Boolean]("m.isMissing"))
    assert(!eval[Boolean]("m.isNotMissing"))

    assert(!eval[Boolean]("a[1].isMissing"))
    assert(eval[Boolean]("a[1].isNotMissing"))

    assert(eval[Boolean]("a[2].isMissing"))
    assert(!eval[Boolean]("a[2].isNotMissing"))

    assert(eval[Int]("as.length") == 2)
    assert(eval[Int]("as[0].a") == 23)
    assert(eval[Boolean]("as[1].b.isMissing"))

    assert(eval[Int]("i") == 5)
    assert(eval[Int]("j") == -7)
    assert(eval[Int]("i.max(j)") == 5)
    assert(eval[Int]("i.min(j)") == -7)
    assert(D_==(eval[Double]("d"), 3.14))
    assert(eval[IndexedSeq[String]]("""s.split(",")""") == IndexedSeq("12", "34", "56", "78"))
    assert(eval[Int]("s2.length") == 62)

    assert(eval[Int]("""a.find(x => x < 0)""") == -1)

    // FIXME catch parse errors
    // assert(eval[Boolean]("i.max(d) == 5"))
  }

  @Test def testParseTypes() {
    val s1 = "SIFT_Score: Double, Age: Int"
    val s2 = ""
    val s3 = "SIFT_Score: Double, Age: Int, SIFT2: BadType"

    assert(Parser.parseAnnotationTypes(s1) == Map("SIFT_Score" -> TDouble, "Age" -> TInt))
    assert(Parser.parseAnnotationTypes(s2) == Map.empty[String, Type])
    intercept[FatalException](Parser.parseAnnotationTypes(s3) == Map("SIFT_Score" -> TDouble, "Age" -> TInt))
  }


  @Test def testTypePretty() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")

    var state = State(sc, sqlContext, vds)
    state = SplitMulti.run(state)
    state = SampleQC.run(state)
    val sb = new StringBuilder

    val weirdSigToAdd = TArray(TStruct(("int1", TInt), ("array1", TArray(TArray(TStruct(
      Array(Field("string1", TString, 0, Map("attr1" -> "I have newlines\nand\t\ttabs", "attr\t2" -> "just some \"quotes\" here")),
        Field("string2", TString, 1))))))))

    val newS = state.vds.vaSignature.insert(weirdSigToAdd, "test", "test2")._1
    newS.pretty(sb, 0, 0, printAttrs = true)
    val res = sb.result()
    val parsed = Parser.parseType(res)

    assert(parsed == newS)

    sb.clear()
    parsed.pretty(sb, 0, 0, printAttrs = false)
    val res2 = sb.result()
    val parsed2 = Parser.parseType(res)

    sb.clear()
    parsed2.pretty(sb, 0, 0, printAttrs = true)
    val res3 = sb.result()
    val parsed3 = Parser.parseType(res)

    assert(parsed2 == parsed3)
  }

  @Test def testJSON() {
    val vds = LoadVCF(sc, "src/test/resources/sample.vcf")

    var state = State(sc, sqlContext, vds)
    state = SplitMulti.run(state)
    state = VariantQC.run(state)

    val va = state.vds.variantsAndAnnotations.take(1).head._2

    val json = state.vds.vaSignature.makeJSON(va)

    val jsonReadBack = VEP.jsonToAnnotation(json, state.vds.vaSignature, "va")
    assert(va == jsonReadBack)
  }
}
