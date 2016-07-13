package org.broadinstitute.hail.methods

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant.Genotype
import org.broadinstitute.hail.{FatalException, SparkSuite}
import org.testng.annotations.Test


class ExprSuite extends SparkSuite {

  @Test def exprTest() {
    val symTab = Map("i" -> (0, TInt),
      "j" -> (1, TInt),
      "d" -> (2, TDouble),
      "d2" -> (3, TDouble),
      "s" -> (4, TString),
      "s2" -> (5, TString),
      "a" -> (6, TArray(TInt)),
      "m" -> (7, TInt),
      "as" -> (8, TArray(TStruct(("a", TInt),
        ("b", TString)))),
      "gs" -> (9, TStruct(("noCall", TGenotype),
        ("homRef", TGenotype),
        ("het", TGenotype),
        ("homVar", TGenotype),
        ("hetNonRef35", TGenotype))),
      "t" -> (10, TBoolean),
      "f" -> (11, TBoolean),
      "mb" -> (12, TBoolean),
      "is" -> (13, TString),
      "iset" -> (14, TSet(TInt)),
      "genedict" -> (15, TDict(TInt)),
      "structArray" -> (16, TArray(TStruct(
        ("f1", TInt),
        ("f2", TString),
        ("f3", TInt)))))
    val ec = EvalContext(symTab)

    val a = ec.a
    a(0) = 5 // i
    a(1) = -7 // j
    a(2) = 3.14
    a(3) = 5.79e7
    a(4) = "12,34,56,78"
    a(5) = "this is a String, there are many like it, but this one is mine"
    a(6) = IndexedSeq(1, 2, null, 6, 3, 3, -1, 8)
    a(7) = null // m
    a(8) = Array[Any](Annotation(23, "foo"), Annotation.empty): IndexedSeq[Any]
    a(9) = Annotation(
      Genotype(),
      Genotype(gt = Some(0)),
      Genotype(gt = Some(1)),
      Genotype(gt = Some(2)),
      Genotype(gt = Some(Genotype.gtIndex(3, 5))))
    a(10) = true
    a(11) = false
    a(12) = null // mb
    a(13) = "-37" // is
    a(14) = Set(0, 1, 2)
    a(15) = Map("gene1" -> 2, "gene2" -> 10, "gene3" -> 14)
    a(16) = IndexedSeq(Annotation(1, "A", 2),
      Annotation(5, "B", 6),
      Annotation(10, "C", 10))

    assert(a.length == symTab.size)

    def eval[T](s: String): Option[T] = {
      val f = Parser.parse(s, ec)._2
      f().map(_.asInstanceOf[T])
    }

    def evalWithType[T](s: String): (BaseType, Option[T]) = {
      val (t, f) = Parser.parse(s, ec)
      (t, f().map(_.asInstanceOf[T]))
    }

    assert(eval[Int]("is.toInt").contains(-37))

    assert(eval[Boolean]("!gs.het.isHomRef").contains(true))

    assert(eval[Boolean]("(1 / 2) == 0.5").contains(true))
    assert(eval[Boolean]("(1.0 / 2.0) == 0.5").contains(true))
    assert(eval[Boolean]("(1 / 2.0) == 0.5").contains(true))
    assert(eval[Boolean]("(1.0 / 2) == 0.5").contains(true))

    assert(eval[Boolean]("isMissing(gs.noCall.gt)").contains(true))
    assert(eval[Boolean]("gs.noCall.gt").isEmpty)

    assert(eval[Boolean]("isMissing(gs.noCall.gtj)").contains(true))
    assert(eval[Boolean]("gs.noCall.gtj").isEmpty)

    assert(eval[Boolean]("isMissing(gs.noCall.gtk)").contains(true))
    assert(eval[Boolean]("gs.noCall.gtk").isEmpty)

    assert(eval[Int]("let a = i and b = j in a + b").contains(-2))
    assert(eval[Int]("let a = i and b = a + j in b").contains(-2))
    assert(eval[Int]("let i = j in i").contains(-7))
    assert(eval[Int]("let a = let b = j in b + 1 in a + 1").contains(-5))

    assert(eval[Boolean]("mb || true").contains(true))
    assert(eval[Boolean]("true || mb").contains(true))
    assert(eval[Boolean]("isMissing(false || mb)").contains(true))
    assert(eval[Boolean]("isMissing(mb || false)").contains(true))

    assert(eval[Int]("gs.homRef.gtj").contains(0)
      && eval[Int]("gs.homRef.gtk").contains(0))
    assert(eval[Int]("gs.het.gtj").contains(0)
      && eval[Int]("gs.het.gtk").contains(1))
    assert(eval[Int]("gs.homVar.gtj").contains(1)
      && eval[Int]("gs.homVar.gtk").contains(1))
    assert(eval[Int]("gs.hetNonRef35.gtj").contains(3)
      && eval[Int]("gs.hetNonRef35.gtk").contains(5))

    assert(eval[Int]("i.orElse(3)").contains(5))
    assert(eval[Int]("m.orElse(3)").contains(3))

    assert(eval[Boolean]("isMissing(i)").contains(false))
    assert(eval[Boolean]("isDefined(i)").contains(true))
    assert(eval[Boolean]("isDefined(i)").contains(true))
    assert(eval[Boolean]("i").nonEmpty)

    assert(eval[Boolean]("isMissing(m)").contains(true))
    assert(eval[Boolean]("isDefined(m)").contains(false))
    assert(eval[Boolean]("m").isEmpty)

    assert(eval[Boolean]("isMissing(a[1])").contains(false))
    assert(eval[Boolean]("isDefined(a[1])").contains(true))
    assert(eval[Boolean]("a[1]").nonEmpty)

    assert(eval[Boolean]("isMissing(a[2])").contains(true))
    assert(eval[Boolean]("isDefined(a[2])").contains(false))
    assert(eval[Boolean]("a[2]").isEmpty)

    assert(eval[Boolean]("1 == 1.0").contains(true))
    val equalsError = intercept[FatalException](eval[Boolean](""" s == 2 """))
    assert(equalsError.getMessage.contains("can only compare objects of similar type"))

    assert(eval[Int]("as.length").contains(2))
    assert(eval[Int]("as[0].a").contains(23))
    assert(eval[Boolean]("isMissing(as[1].b)").contains(true))
    assert(eval[Boolean]("as[1].b").isEmpty)

    assert(eval[Int]("i").contains(5))
    assert(eval[Int]("j").contains(-7))
    assert(eval[Int]("i.max(j)").contains(5))
    assert(eval[Int]("i.min(j)").contains(-7))
    assert(eval[Double]("d").exists(D_==(_, 3.14)))
    assert(eval[IndexedSeq[String]]("""s.split(",")""").contains(IndexedSeq("12", "34", "56", "78")))
    assert(eval[Int]("s2.length").contains(62))

    assert(eval[Int]("""a.find(x => x < 0)""").contains(-1))

    assert(eval[String](""" "HELLO=" + j + ", asdasd" + 9""")
      .contains("HELLO=-7, asdasd9"))

    assert(eval[IndexedSeq[_]](""" a.filter(x => x < 4)   """)
      .contains(IndexedSeq(1, 2, 3, 3, -1)))

    assert(eval[IndexedSeq[_]](""" a.filter(x => x < 4).map(x => x * 100)   """)
      .contains(IndexedSeq(1, 2, 3, 3, -1).map(_ * 100)))

    assert(eval[Boolean](""" a.filter(x => x < 4).map(x => x * 100).exists(x => x == -100)   """)
      .contains(true))

    assert(eval[Int]("""a.min""").contains(-1))
    assert(eval[Int]("""a.max""").contains(8))
    assert(eval[Int]("""a.sum""").contains(IndexedSeq(1, 2, 6, 3, 3, -1, 8).sum))
    assert(eval[String]("""str(i)""").contains("5"))
    assert(eval[String](""" 5 + "5" """) == eval[String](""" "5" + 5 """))
    assert(eval[Int]("""iset.min""").contains(0))
    assert(eval[Int]("""iset.max""").contains(2))
    assert(eval[Int]("""iset.sum""").contains(3))

    assert(eval[String](""" "a b c d".replace(" ", "_") """).contains("a_b_c_d"))
    assert(eval[String](" \"a\\tb\".replace(\"\\t\", \"_\") ").contains("a_b"))
    assert(eval[String](""" "a    b  c    d".replace("\\s+", "_") """).contains("a_b_c_d"))

    assert(eval[String]("""NA: String""").isEmpty)
    assert(eval[String]("""NA: Int""").isEmpty)
    assert(eval[String]("""NA: Array[Int]""").isEmpty)

    assert(eval[IndexedSeq[Any]]("""[1, 2, 3, 4]""").contains(IndexedSeq(1, 2, 3, 4)))
    assert(eval[IndexedSeq[Any]]("""[1, 2, NA:Int, 6, 3, 3, -1, 8]""").contains(IndexedSeq(1, 2, null, 6, 3, 3, -1, 8)))


    assert(eval[IndexedSeq[Any]]("""[1, 2, 3.0, 4]""").contains(IndexedSeq(1, 2, 3.0, 4)))
    assert(eval[Double]("""[1, 2, 3.0, 4].max""").contains(4.0))

    intercept[FatalException](eval[IndexedSeq[Any]]("""[1,2, "hello"] """))
    intercept[FatalException](eval[IndexedSeq[Any]]("""[] """))

    val (t, r) = evalWithType[Annotation](""" {"field1": 1, "field2": 2 } """)
    assert(r.contains(Annotation(1, 2)))
    assert(t == TStruct(("field1", TInt), ("field2", TInt)))

    val (t2, r2) = evalWithType[Annotation](""" {"field1": 1, "asdasd": "Hello" } """)
    assert(r2.contains(Annotation(1, "Hello")))
    assert(t2 == TStruct(("field1", TInt), ("asdasd", TString)))

    assert(eval[IndexedSeq[_]](""" [0,1,2,3][0:2] """).contains(IndexedSeq(0, 1)))
    assert(eval[IndexedSeq[_]](""" [0,1,2,3][2:] """).contains(IndexedSeq(2, 3)))
    assert(eval[IndexedSeq[_]](""" [0,1,2,3][:2] """).contains(IndexedSeq(0, 1)))
    assert(eval[IndexedSeq[_]](""" [0,1,2,3][:] """).contains(IndexedSeq(0, 1, 2, 3)))

    assert(eval[Int](""" genedict["gene2"] """).contains(10))

    val (dictType, result) = evalWithType[Map[_, _]](""" index(structArray, f2) """)
    assert(result.contains(
      Map("A" -> Annotation(1, 2),
        "B" -> Annotation(5, 6),
        "C" -> Annotation(10, 10))
    ))
    assert(dictType == TDict(TStruct(("f1", TInt), ("f3", TInt))))


    assert(eval[Int](""" index(structArray, f2)["B"].f3 """).contains(6))
    assert(eval[Map[_, _]](""" index(structArray, f2).mapvalues(x => x.f1) """).contains(Map(
      "A" -> 1,
      "B" -> 5,
      "C" -> 10)
    ))
    assert(eval[Boolean](""" index(structArray, f2).contains("B") """).contains(true))
    assert(eval[Boolean](""" index(structArray, f2).contains("E") """).contains(false))
    // FIXME catch parse errors
  }

  @Test def testParseTypes() {
    val s1 = "SIFT_Score: Double, Age: Int"
    val s2 = ""
    val s3 = "SIFT_Score: Double, Age: Int, SIFT2: BadType"

    assert(Parser.parseAnnotationTypes(s1) == Map("SIFT_Score" -> TDouble, "Age" -> TInt))
    assert(Parser.parseAnnotationTypes(s2) == Map.empty[String, BaseType])
    intercept[FatalException](Parser.parseAnnotationTypes(s3) == Map("SIFT_Score" -> TDouble, "Age" -> TInt))
  }

  @Test def testTypePretty() {
    import Type._
    // for arbType

    val sb = new StringBuilder
    check(forAll { (t: Type) =>
      sb.clear()
      t.pretty(sb, 0)
      val res = sb.result()
      val parsed = Parser.parseType(res)
      t == parsed
    })
  }

  @Test def testJSON() {
    check(forAll { (t: Type) =>
      val a = t.genValue.sample()
      val json = t.toJSON(a)
      a == JSONAnnotationImpex.importAnnotation(json, t)
    })
  }

  @Test def testReadWrite() {
    check(forAll { (t: Type) =>
      val sb = new StringBuilder
      t.pretty(sb, 0)
      val a = t.genValue.sample()
      JSONAnnotationImpex.importAnnotation(
        JSONAnnotationImpex.exportAnnotation(a, t), t) == a
    })
  }

}
