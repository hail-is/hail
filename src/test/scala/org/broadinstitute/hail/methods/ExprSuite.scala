package org.broadinstitute.hail.methods

import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.check.Properties
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.utils.{FatalException, Interval}
import org.broadinstitute.hail.utils.StringEscapeUtils._
import org.broadinstitute.hail.variant.{Genotype, Locus, Variant}
import org.broadinstitute.hail.{SparkSuite, TestUtils}
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.testng.annotations.Test
import org.broadinstitute.hail.TestUtils._

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
        ("f3", TInt)))),
      "a2" -> (17, TArray(TString)),
      "nullarr" -> (18, TArray(TInt)),
      "nullset" -> (19, TSet(TInt)),
      "emptyarr" -> (20, TArray(TInt)),
      "emptyset" -> (21, TSet(TInt)))

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
    a(17) = IndexedSeq("a", "d", null, "c", "e", null, "d", "c")
    a(18) = null
    a(19) = null
    a(20) = IndexedSeq[Int]()
    a(21) = Set[Int]()

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

    assert(eval[IndexedSeq[_]]("""a.flatMap(x => [x])""").contains(IndexedSeq(1, 2, null, 6, 3, 3, -1, 8)))
    assert(eval[IndexedSeq[_]]("""a.flatMap(x => [x, x + 1])""").contains(IndexedSeq(1, 2, 2, 3, null, null, 6, 7, 3, 4, 3, 4, -1, 0, 8, 9)))

    assert(eval[IndexedSeq[_]]("""nullarr.flatMap(x => [x])""").isEmpty)
    assert(eval[IndexedSeq[_]]("""emptyarr.flatMap(x => [x])""").contains(IndexedSeq[Int]()))
    assert(eval[IndexedSeq[_]]("""emptyarr.flatMap(x => nullarr)""").contains(IndexedSeq[Int]()))
    assert(eval[IndexedSeq[_]]("""a.flatMap(x => nullarr)""").isEmpty)
    assert(eval[IndexedSeq[_]]("""[nullarr, [1], [2]].flatMap(x => x)""").isEmpty)
    assert(eval[IndexedSeq[_]]("""[[0], nullarr, [2]].flatMap(x => x)""").isEmpty)
    assert(eval[IndexedSeq[_]]("""[[0], [1], nullarr].flatMap(x => x)""").isEmpty)

    assert(eval[Set[_]]("""iset.flatMap(x => [x].toSet)""").contains(Set(0, 1, 2)))
    assert(eval[Set[_]]("""iset.flatMap(x => [x, x + 1].toSet)""").contains(Set(0, 1, 2, 3)))

    assert(eval[Set[_]]("""nullset.flatMap(x => [x].toSet)""").isEmpty)
    assert(eval[Set[_]]("""emptyset.flatMap(x => [x].toSet)""").contains(Set[Int]()))
    assert(eval[Set[_]]("""emptyset.flatMap(x => nullset)""").contains(Set[Int]()))
    assert(eval[Set[_]]("""iset.flatMap(x => nullset)""").isEmpty)
    assert(eval[Set[_]]("""[nullset, [1].toSet, [2].toSet].toSet.flatMap(x => x)""").isEmpty)
    assert(eval[Set[_]]("""[[0].toSet, nullset, [2].toSet].toSet.flatMap(x => x)""").isEmpty)
    assert(eval[Set[_]]("""[[0].toSet, [1].toSet, nullset].toSet.flatMap(x => x)""").isEmpty)

    assert(eval[Set[_]]("""[[0].toSet, [1].toSet, nullset].filter(s => isDefined(s)).toSet.flatMap(x => x)""").contains(Set(0, 1)))

    TestUtils.interceptFatal("""expects a lambda function""")(
      eval[Set[_]]("""iset.flatMap(0)"""))

    TestUtils.interceptFatal("""expects lambda body to have type Array\[T\] or Set\[T\], got Int""")(
      eval[Set[_]]("""iset.flatMap(x => x)"""))

    TestUtils.interceptFatal("""match, got Set\[Int\] and Array\[Int\]""")(
      eval[Set[_]]("""iset.flatMap(x => [x])"""))

    TestUtils.interceptFatal("""match, got Array\[Int\] and Set\[Int\]""")(
      eval[IndexedSeq[_]]("""a.flatMap(x => [x].toSet)"""))

    assert(eval[IndexedSeq[_]](""" [[1], [2, 3], [4, 5, 6]].flatten() """).contains(IndexedSeq(1, 2, 3, 4, 5, 6)))
    assert(eval[IndexedSeq[_]](""" [a, [1]].flatten() """).contains(IndexedSeq(1, 2, null, 6, 3, 3, -1, 8, 1)))

    assert(eval[Set[_]](""" [[1], [2]].flatten().toSet """).contains(Set(1, 2)))

    assert(eval[IndexedSeq[_]](""" [nullarr].flatten() """).isEmpty)
    assert(eval[IndexedSeq[_]](""" [[0], nullarr].flatten() """).isEmpty)
    assert(eval[IndexedSeq[_]](""" [nullarr, [1]].flatten() """).isEmpty)

    assert(eval[IndexedSeq[_]](""" [[1], nullarr, [2, 3]].filter(a => isDefined(a)).flatten() """).contains(IndexedSeq(1, 2, 3)))

    assert(eval[Set[_]](""" [iset, [2, 3, 4].toSet].toSet.flatten() """).contains(Set(0, 1, 2, 3, 4)))

    assert(eval[Set[_]](""" [nullset].toSet.flatten() """).isEmpty)
    assert(eval[Set[_]](""" [[0].toSet, nullset].toSet.flatten() """).isEmpty)
    assert(eval[Set[_]](""" [nullset, [1].toSet].toSet.flatten() """).isEmpty)

    TestUtils.interceptFatal("""expects type Array\[Array\[T\]\] or Set\[Set\[T\]\], got Array\[Set\[Int\]\]""")(
      eval[Set[_]](""" [iset, [2, 3, 4].toSet].flatten() """))

    TestUtils.interceptFatal("""expects type Array\[Array\[T\]\] or Set\[Set\[T\]\], got Set\[Array\[Int\]\]""")(
      eval[Set[_]](""" [[1], [2, 3, 4]].toSet.flatten() """))

    TestUtils.interceptFatal("""expects type Array\[Array\[T\]\] or Set\[Set\[T\]\], got Array\[Int\]""")(
      eval[Set[_]](""" [0].flatten() """))

    TestUtils.interceptFatal("""does not take parameters, use flatten()""")(
      eval[Set[_]](""" [[0]].flatten(0) """))


    assert(eval[IndexedSeq[_]]("""a.sort()""").contains(IndexedSeq(-1, 1, 2, 3, 3, 6, 8, null)))
    assert(eval[IndexedSeq[_]]("""a.sort(true)""").contains(IndexedSeq(-1, 1, 2, 3, 3, 6, 8, null)))
    assert(eval[IndexedSeq[_]]("""a.sort(false)""").contains(IndexedSeq(8, 6, 3, 3, 2, 1, -1, null)))

    assert(eval[IndexedSeq[_]]("""a2.sort()""").contains(IndexedSeq("a", "c", "c", "d", "d", "e", null, null)))
    assert(eval[IndexedSeq[_]]("""a2.sort(true)""").contains(IndexedSeq("a", "c", "c", "d", "d", "e", null, null)))
    assert(eval[IndexedSeq[_]]("""a2.sort(false)""").contains(IndexedSeq("e", "d", "d", "c", "c", "a", null, null)))

    TestUtils.interceptFatal("""expects at most one Boolean parameter""")(
      eval[IndexedSeq[_]]("""a.sort(0)"""))
    TestUtils.interceptFatal("""expects at most one Boolean parameter""")(
      eval[IndexedSeq[_]]("""a.sort("asdasd")"""))
    TestUtils.interceptFatal("""expects at most one Boolean parameter""")(
      eval[IndexedSeq[_]]("""a.sort(true, true)"""))
    TestUtils.interceptFatal("""expects at most one Boolean parameter""")(
      eval[IndexedSeq[_]]("""a.sort(null)"""))
    TestUtils.interceptFatal("""expects at most one Boolean parameter""")(
      eval[IndexedSeq[_]]("""a.sort(asdasd)"""))


    assert(eval[IndexedSeq[_]]("""a.sortBy(x => x)""").contains(IndexedSeq(-1, 1, 2, 3, 3, 6, 8, null)))
    assert(eval[IndexedSeq[_]]("""a.sortBy(x => -x)""").contains(IndexedSeq(8, 6, 3, 3, 2, 1, -1, null)))
    assert(eval[IndexedSeq[_]]("""a.sortBy(x => (x - 2) * (x + 1))""").contains(IndexedSeq(1, 2, -1, 3, 3, 6, 8, null)))

    assert(eval[IndexedSeq[_]]("""a.sortBy(x => x, true)""").contains(IndexedSeq(-1, 1, 2, 3, 3, 6, 8, null)))
    assert(eval[IndexedSeq[_]]("""a.sortBy(x => x, false)""").contains(IndexedSeq(8, 6, 3, 3, 2, 1, -1, null)))

    assert(eval[IndexedSeq[_]]("""a2.sortBy(x => x)""").contains(IndexedSeq("a", "c", "c", "d", "d", "e", null, null)))
    assert(eval[IndexedSeq[_]]("""a2.sortBy(x => x, true)""").contains(IndexedSeq("a", "c", "c", "d", "d", "e", null, null)))
    assert(eval[IndexedSeq[_]]("""a2.sortBy(x => x, false)""").contains(IndexedSeq("e", "d", "d", "c", "c", "a", null, null)))

    TestUtils.interceptFatal("""lambda function \(param => T\) and at most one Boolean parameter""")(
      eval[IndexedSeq[_]]("""a.sortBy(0)"""))
    TestUtils.interceptFatal("""lambda function \(param => T\) and at most one Boolean parameter""")(
      eval[IndexedSeq[_]]("""a.sortBy(0, true)"""))
    TestUtils.interceptFatal("""lambda function \(param => T\) and at most one Boolean parameter""")(
      eval[IndexedSeq[_]]("""a.sortBy(x => x, 0)"""))
    TestUtils.interceptFatal("""lambda function \(param => T\) and at most one Boolean parameter""")(
      eval[IndexedSeq[_]]("""a.sortBy(x => x, "asdasd")"""))
    TestUtils.interceptFatal("""lambda function \(param => T\) and at most one Boolean parameter""")(
      eval[IndexedSeq[_]]("""a.sortBy(x => x, true, true)"""))
    TestUtils.interceptFatal("""lambda function \(param => T\) and at most one Boolean parameter""")(
      eval[IndexedSeq[_]]("""a.sortBy(x => x, null)"""))
    TestUtils.interceptFatal("""lambda function \(param => T\) and at most one Boolean parameter""")(
      eval[IndexedSeq[_]]("""a.sortBy(x => x, asdasd)"""))

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

    assert(eval[String](""" "\t\t\t" """).contains("\t\t\t"))
    assert(eval[String](""" "\"\"\"" """).contains("\"\"\""))
    assert(eval[String](""" "```" """).contains("```"))

    assert(eval[String](""" "\t" """) == eval[String]("\"\t\""))

    assert(eval[String](""" "a b c d".replace(" ", "_") """).contains("a_b_c_d"))
    assert(eval[String](" \"a\\tb\".replace(\"\\t\", \"_\") ").contains("a_b"))
    assert(eval[String](""" "a    b  c    d".replace("\\s+", "_") """).contains("a_b_c_d"))

    // quoting '` optional in strings
    assert(eval[String](""" "\"\'\`" """).contains(""""'`"""))
    assert(eval[String](""" "\"'`" """).contains(""""'`"""))

    // quoting "' optional in literal identifiers
    assert(eval[Int]("""let `\"\'\`` = 5 in `"'\``""").contains(5))

    assert(eval[String]("""NA: String""").isEmpty)
    assert(eval[String]("""NA: Int""").isEmpty)
    assert(eval[String]("""NA: Array[Int]""").isEmpty)

    assert(eval[IndexedSeq[Any]]("""[1, 2, 3, 4]""").contains(IndexedSeq(1, 2, 3, 4)))
    assert(eval[IndexedSeq[Any]]("""[1, 2, NA:Int, 6, 3, 3, -1, 8]""").contains(IndexedSeq(1, 2, null, 6, 3, 3, -1, 8)))


    assert(eval[IndexedSeq[Any]]("""[1, 2, 3.0, 4]""").contains(IndexedSeq(1, 2, 3.0, 4)))
    assert(eval[Double]("""[1, 2, 3.0, 4].max""").contains(4.0))

    intercept[FatalException](eval[IndexedSeq[Any]]("""[1,2, "hello"] """))
    intercept[FatalException](eval[IndexedSeq[Any]]("""[] """))

    val (t, r) = evalWithType[Annotation](""" {field1: 1, field2: 2 } """)
    assert(r.contains(Annotation(1, 2)))
    assert(t == TStruct(("field1", TInt), ("field2", TInt)))

    val (t2, r2) = evalWithType[Annotation](""" {field1: 1, asdasd: "Hello" } """)
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
    assert(eval[Map[_, _]](""" index(structArray, f2).mapValues(x => x.f1) """).contains(Map(
      "A" -> 1,
      "B" -> 5,
      "C" -> 10)
    ))
    assert(eval[Boolean](""" index(structArray, f2).contains("B") """).contains(true))
    assert(eval[Boolean](""" index(structArray, f2).contains("E") """).contains(false))

    assert(eval[Map[_, _]](""" genedict.mapValues(x => x + 1) """).contains(Map("gene1" -> 3, "gene2" -> 11, "gene3" -> 15)))

    // caused exponential blowup previously
    assert(eval[Boolean](
      """
        |if (false) false
        |else if (false) false
        |else if (false) false
        |else if (false) false
        |else if (false) false
        |else if (false) false
        |else if (false) false
        |else if (false) false
        |else if (false) false
        |else if (false) false
        |else if (false) false
        |else if (false) false
        |else if (false) false
        |else if (false) false
        |else if (false) false
        |else if (false) false
        |else if (false) false
        |else if (false) false
        |else if (false) false
        |else if (false) false
        |else if (false) false
        |else if (false) false
        |else if (false) false
        |else if (false) false
        |else if (false) false
        |else true
      """.stripMargin).contains(true))

    assert(eval[Annotation]("""merge({a: 1, b: 2}, {c: false, d: true}) """).contains(Annotation(1, 2, false, true)))
    assert(eval[Annotation]("""merge(NA: Struct{a: Int, b: Int}, {c: false, d: true}) """).contains(Annotation(null, null, false, true)))
    assert(eval[Annotation]("""merge({a: 1, b: 2}, NA: Struct{c: Boolean, d: Boolean}) """).contains(Annotation(1, 2, null, null)))
    assert(eval[Annotation]("""merge(NA: Struct{a: Int, b: Int}, NA: Struct{c: Boolean, d: Boolean}) """).isEmpty)
    TestUtils.interceptFatal("invalid merge operation: same-name fields")(
      eval[Annotation]("""merge({a: 1, b: 2}, {c: false, d: true, a: 1, b: 0}) """).contains(Annotation(1, 2, false, true)))
    TestUtils.interceptFatal("invalid arguments to `merge'")(
      eval[Annotation]("""merge(NA: Struct{a: Int, b: Int}) """).isEmpty)
    TestUtils.interceptFatal("invalid arguments to `merge'")(
      eval[Annotation]("""merge(NA: Struct{a: Int, b: Int}, 5) """).isEmpty)

    assert(eval[Annotation](""" select({a:1,b:2}, a) """).contains(Annotation(1)))
    assert(eval[Boolean](""" let x = {a:1, b:2, c:3, `\tweird\t`: 4} in select(x, a,b,`\tweird\t`) == drop(x, c) """).contains(true))
    TestUtils.interceptFatal("too few arguments for method `select'")(
      eval[Annotation](""" let x = {a:1, b:2, c:3, `\tweird\t`: 4} in select(x) """))
    TestUtils.interceptFatal("invalid arguments for method `select'")(
      eval[Annotation](""" let x = {a:1, b:2, c:3, `\tweird\t`: 4} in select(x, 5,6,7) """))
    TestUtils.interceptFatal("invalid arguments for method `select'\\s+Duplicate identifiers found")(
      eval[Annotation](""" let x = {a:1, b:2, c:3, `\tweird\t`: 4} in select(x, a,a,b,c,c) """))
    TestUtils.interceptFatal("invalid arguments for method `select'\\s+invalid struct filter operation:\\s+fields \\[ ., . \\] not found")(
      eval[Annotation](""" let x = {a:1, b:2, c:3, `\tweird\t`: 4} in select(x, a,b,c,d,e) """))

    assert(eval[Annotation](""" drop({a:1,b:2}, a) """).contains(Annotation(2)))
    TestUtils.interceptFatal("too few arguments for method `drop'")(
      eval[Annotation](""" let x = {a:1, b:2, c:3, `\tweird\t`: 4} in drop(x) """))
    TestUtils.interceptFatal("invalid arguments for method `drop'")(
      eval[Annotation](""" let x = {a:1, b:2, c:3, `\tweird\t`: 4} in drop(x, 5,6,7) """))
    TestUtils.interceptFatal("invalid arguments for method `drop'\\s+Duplicate identifiers found")(
      eval[Annotation](""" let x = {a:1, b:2, c:3, `\tweird\t`: 4} in drop(x, a,a,b,c,c) """))
    TestUtils.interceptFatal("invalid arguments for method `drop'\\s+invalid struct filter operation:\\s+fields \\[ ., . \\] not found")(
      eval[Annotation](""" let x = {a:1, b:2, c:3, `\tweird\t`: 4} in drop(x, a,b,c,d,e) """))

    assert(eval[Variant]("""Variant("1", 1, "A", "T")""").contains(Variant("1", 1, "A", "T")))
    assert(eval[Variant]("""Variant("1", 1, "A", ["T", "G"])""").contains(Variant("1", 1, "A", Array("T", "G"))))
    assert(eval[Boolean]("""let v = Variant("1", 1, "A", "T") in Variant(str(v)) == v""").contains(true))

    assert(eval[Locus]("""Locus("1", 1)""").contains(Locus("1", 1)))
    assert(eval[Locus]("""Locus("1:1")""").contains(Locus("1", 1)))
    assert(eval[Boolean]("""let l = Locus("1", 1) in Locus(str(l)) == l""").contains(true))

    assert(eval[Interval[Locus]]("""Interval(Locus("1", 1), Locus("2", 2))""").contains(Interval(Locus("1", 1), Locus("2", 2))))
    assert(eval[Locus](""" Interval(Locus("1", 1), Locus("2", 2)).start """).contains(Locus("1", 1)))
    assert(eval[Locus](""" Interval(Locus("1", 1), Locus("2", 2)).end """).contains(Locus("2", 2)))
    assert(eval[Boolean]("""Interval(Locus("1", 1), Locus("1", 3)).contains(Locus("1", 2)) """).contains(true))
    assert(eval[Boolean]("""Interval(Locus("1", 1), Locus("1", 3)).contains(Locus("2", 2)) """).contains(false))

    // FIXME catch parse errors
    assert(eval(""" "\``\''" """) == eval(""" "``''" """))
    TestUtils.interceptFatal("""invalid escape character.*string.*\\a""")(eval[String](""" "this is bad \a" """))
    TestUtils.interceptFatal("""unterminated string literal""")(eval[String](""" "unclosed string \" """))
    TestUtils.interceptFatal("""invalid escape character.*backtick identifier.*\\i""")(eval[String](""" let `bad\identifier` = 0 in 0 """))
    TestUtils.interceptFatal("""unterminated backtick identifier""")(eval[String](""" let `bad\identifier = 0 in 0 """))

    assert(D_==(eval[Double]("log(56.toLong)").get, math.log(56)))
    assert(D_==(eval[Double]("exp(5.6)").get, math.exp(5.6)))
    assert(D_==(eval[Double]("log10(5.6)").get, math.log10(5.6)))
    assert(D_==(eval[Double]("log10(5.6)").get, eval[Double]("log(5.6, 10)").get))
    assert(D_==(eval[Double]("log(5.6, 3.2)").get, 1.481120576298196))
    assert(D_==(eval[Double]("sqrt(5.6)").get, math.sqrt(5.6)))
    assert(D_==(eval[Double]("pow(2, 3)").get, 8.0))

    assert(eval[IndexedSeq[Int]]("""[1,2,3] + [2,3,4] """).contains(IndexedSeq(3, 5, 7)))
    assert(eval[IndexedSeq[Int]]("""[1,2,3] - [2,3,4] """).contains(IndexedSeq(-1, -1, -1)))
    assert(eval[IndexedSeq[Double]]("""[1,2,3] / [2,3,4] """).contains(IndexedSeq(.5, 2.toDouble / 3.toDouble, .75)))
    assert(eval[IndexedSeq[Double]]("""1 / [2,3,4] """).contains(IndexedSeq(1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0)))
    assert(eval[IndexedSeq[Double]]("""[2,3,4] / 2""").contains(IndexedSeq(1.0, 1.5, 2.0)))
    assert(eval[IndexedSeq[Double]]("""[2,3,4] * 2.0""").contains(IndexedSeq(4.0, 6.0, 8.0)))
    assert(eval[IndexedSeq[_]]("""[2,NA: Int,4] * 2.0""").contains(IndexedSeq(4.0, null, 8.0)))
    assert(eval[IndexedSeq[_]]("""[2,NA: Int,4] * [2,NA: Int,4]""").contains(IndexedSeq(4.0, null, 16.0)))
    assert(eval[IndexedSeq[_]]("""[2,NA: Int,4] + 2.0""").contains(IndexedSeq(4.0, null, 6.0)))
    assert(eval[IndexedSeq[_]]("""[2,NA: Int,4] + [2,NA: Int,4]""").contains(IndexedSeq(4.0, null, 8.0)))
    assert(eval[IndexedSeq[_]]("""[2,NA: Int,4] - 2.0""").contains(IndexedSeq(0.0, null, 2.0)))
    assert(eval[IndexedSeq[_]]("""[2,NA: Int,4] - [2,NA: Int,4]""").contains(IndexedSeq(0.0, null, 0.0)))
    assert(eval[IndexedSeq[_]]("""[2,NA: Int,4] / 2.0""").contains(IndexedSeq(1.0, null, 2.0)))
    assert(eval[IndexedSeq[_]]("""[2,NA: Int,4] / [2,NA: Int,4]""").contains(IndexedSeq(1.0, null, 1.0)))

    TestUtils.interceptFatal("""cannot apply operation `\+' to arrays of unequal length""") {
      eval[IndexedSeq[Int]]("""[1] + [2,3,4] """)
    }

    interceptFatal("No function found") {
      eval[Double](""" log(Variant("22", 123, "A", "T")) """)
    }

    assert(eval[Int]("[0, 0.toLong][0].toInt").contains(0))
    assert(eval[Int]("[0, 0.toFloat][0].toInt").contains(0))
    assert(eval[Int]("[0, 0.toDouble][0].toInt").contains(0))
    assert(eval[Boolean]("[NA:Int] == [0]").contains(false))
    assert(eval[Boolean]("[NA:Long] == [0.toLong]").contains(false))
    assert(eval[Boolean]("[NA:Float] == [0.toFloat]").contains(false))
    assert(eval[Boolean]("[NA:Double] == [0.toDouble]").contains(false))

    assert(eval[IndexedSeq[Int]]("range(5)").contains(IndexedSeq(0, 1, 2, 3, 4)))
    assert(eval[IndexedSeq[Int]]("range(1)").contains(IndexedSeq(0)))
    assert(eval[IndexedSeq[Int]]("range(10, 14)").contains(IndexedSeq(10, 11, 12, 13)))
    assert(eval[IndexedSeq[Int]]("range(-2, 2)").contains(IndexedSeq(-2, -1, 0, 1)))

    assert(eval[Boolean]("pcoin(2.0)").contains(true))
    assert(eval[Boolean]("pcoin(-1.0)").contains(false))
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
      t.pretty(sb, compact = true, printAttrs = true)
      val res = sb.result()
      val parsed = Parser.parseType(res)
      t == parsed
    })
    check(forAll { (t: Type) =>
      sb.clear()
      t.pretty(sb, printAttrs = true)
      val res = sb.result()
      //      println(res)
      val parsed = Parser.parseType(res)
      t == parsed
    })

  }

  @Test def testEscaping() {
    val p = forAll { (s: String) =>
      s == unescapeString(escapeString(s))
    }

    p.check()
  }

  @Test def testEscapingSimple() {
    // a == 0x61, _ = 0x5f
    assert(escapeStringSimple("abc", '_', _ => false) == "abc")
    assert(escapeStringSimple("abc", '_', _ == 'a') == "_61bc")
    assert(escapeStringSimple("abc_", '_', _ => false) == "abc_5f")
    assert(unescapeStringSimple("abc", '_') == "abc")
    assert(unescapeStringSimple("abc_5f", '_') == "abc_")
    assert(unescapeStringSimple("_61bc", '_') == "abc")
    assert(unescapeStringSimple("_u0061bc", '_') == "abc")
    assert(escapeStringSimple("my name is 名谦", '_', _ => false) == "my name is _u540d_u8c26")
    assert(unescapeStringSimple("my name is _u540d_u8c26", '_') == "my name is 名谦")

    val p = forAll { (s: String) =>
      s == unescapeStringSimple(escapeStringSimple(s, '_', _.isLetterOrDigit, _.isLetterOrDigit), '_')
    }

    p.check()
  }

  @Test def testImpexes() {

    val g = for {t <- Type.genArb
      a <- t.genValue} yield (t, a)

    object Spec extends Properties("ImpEx") {
      property("json") = forAll(g) { case (t, a) =>
        JSONAnnotationImpex.importAnnotation(JSONAnnotationImpex.exportAnnotation(a, t), t) == a
      }

      property("json-text") = forAll(g) { case (t, a) =>
        val string = compact(JSONAnnotationImpex.exportAnnotation(a, t))
        JSONAnnotationImpex.importAnnotation(parse(string), t) == a
      }

      property("table") = forAll(g.filter { case (t, a) => t != TDouble && a != null }) { case (t, a) =>
        TableAnnotationImpex.importAnnotation(TableAnnotationImpex.exportAnnotation(a, t), t) == a
      }

      property("spark") = forAll(g) { case (t, a) =>
        SparkAnnotationImpex.importAnnotation(SparkAnnotationImpex.exportAnnotation(a, t), t) == a
      }
    }

    Spec.check()
  }

  @Test def testIfNumericPromotion() {
    val ec = EvalContext(Map("c" -> (0, TBoolean), "l" -> (1, TLong), "f" -> (2, TFloat)))
    def eval[T](s: String): (BaseType, Option[T]) = {
      val (t, f) = Parser.parse(s, ec)
      (t, f().map(_.asInstanceOf[T]))
    }

    assert(Parser.parse("if (c) 0 else 0", ec)._1 == TInt)
    assert(Parser.parse("if (c) 0 else l", ec)._1 == TLong)
    assert(Parser.parse("if (c) f else 0", ec)._1 == TFloat)
    assert(Parser.parse("if (c) 0 else 0.0", ec)._1 == TDouble)
    assert(eval[Int]("(if (true) 0 else 0.toLong).toInt") == (TInt, Some(0)))
    assert(eval[Int]("(if (true) 0 else 0.toFloat).toInt") == (TInt, Some(0)))
  }
}
