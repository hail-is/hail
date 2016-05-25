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
      "mb" ->(12, TBoolean),
      "is" ->(13, TString),
      "iset" ->(14, TSet(TInt)))
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
    assert(a.length == 15)

    def eval[T](s: String): Option[T] = {
      val f = Parser.parse(s, ec)._2
      f().map(_.asInstanceOf[T])
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

    // FIXME catch parse errors
    // assert(eval[Boolean]("i.max(d) == 5"))
  }

  @Test def testAssign() {
    val t1 = TStruct.empty

    val (t2, insb) = t1.insert(TInt, "a", "b")
    val (t3, insc) = t2.insert(TDouble, "a", "c")

    val (tb, assb) = t3.assign("a", "b")
    assert(tb == TInt)

    val (tc, assc) = t3.assign("a", "c")
    assert(tc == TDouble)

    val qc = t3.query("a", "c")

    val v1 = Annotation.empty
    val v2 = insb(v1, Some(5))
    val v3 = insc(v2, Some(7.2))

    assert(qc(assc(v3, Some(-3.2))).contains(-3.2))
    assert(qc(assc(v3, None)).isEmpty)

    val v5 = assc(Annotation.empty, Some(6.7))

    assert(qc(v5).contains(6.7))
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
      val json = t.makeJSON(a)
      a == Annotation.fromJson(json, t, "")
    })
  }

  @Test def testReadWrite() {
    check(forAll { (t: Type) =>
      val sb = new StringBuilder
      t.pretty(sb, 0)
      val a = t.genValue.sample()
      t.makeSparkReadable(t.makeSparkWritable(a)) == a
    })
  }

}
