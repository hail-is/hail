package is.hail.variant

import is.hail.{HailSuite, TestUtils}
import is.hail.utils._
import org.testng.annotations.Test

class LocusIntervalSuite extends HailSuite {
  def rg = ctx.getReference(ReferenceGenome.GRCh37)

  @Test def testParser() {
    val xMax = rg.contigLength("X")
    val yMax = rg.contigLength("Y")
    val chr22Max = rg.contigLength("22")

    assert(Locus.parseInterval("[1:100-1:101)", rg) == Interval(Locus("1", 100), Locus("1", 101), true, false))
    assert(Locus.parseInterval("[1:100-101)", rg) == Interval(Locus("1", 100), Locus("1", 101), true, false))
    assert(Locus.parseInterval("[X:100-101)", rg) == Interval(Locus("X", 100), Locus("X", 101), true, false))
    assert(Locus.parseInterval("[X:100-end)", rg) == Interval(Locus("X", 100), Locus("X", xMax), true, false))
    assert(Locus.parseInterval("[X:100-End)", rg) == Interval(Locus("X", 100), Locus("X", xMax), true, false))
    assert(Locus.parseInterval("[X:100-END)", rg) == Interval(Locus("X", 100), Locus("X", xMax), true, false))
    assert(Locus.parseInterval("[X:start-101)", rg) == Interval(Locus("X", 1), Locus("X", 101), true, false))
    assert(Locus.parseInterval("[X:Start-101)", rg) == Interval(Locus("X", 1), Locus("X", 101), true, false))
    assert(Locus.parseInterval("[X:START-101)", rg) == Interval(Locus("X", 1), Locus("X", 101), true, false))
    assert(Locus.parseInterval("[X:START-Y:END)", rg) == Interval(Locus("X", 1), Locus("Y", yMax), true, false))
    assert(Locus.parseInterval("[X-Y)", rg) == Interval(Locus("X", 1), Locus("Y", yMax), true, false))
    assert(Locus.parseInterval("[1-22)", rg) == Interval(Locus("1", 1), Locus("22", chr22Max), true, false))

    assert(Locus.parseInterval("1:100-1:101", rg) == Interval(Locus("1", 100), Locus("1", 101), true, false))
    assert(Locus.parseInterval("1:100-101", rg) == Interval(Locus("1", 100), Locus("1", 101), true, false))
    assert(Locus.parseInterval("X:100-end", rg) == Interval(Locus("X", 100), Locus("X", xMax), true, true))
    assert(Locus.parseInterval("(X:100-End]", rg) == Interval(Locus("X", 100), Locus("X", xMax), false, true))
    assert(Locus.parseInterval("(X:100-END)", rg) == Interval(Locus("X", 100), Locus("X", xMax), false, false))
    assert(Locus.parseInterval("[X:start-101)", rg) == Interval(Locus("X", 1), Locus("X", 101), true, false))
    assert(Locus.parseInterval("(X:Start-101]", rg) == Interval(Locus("X", 1), Locus("X", 101), false, true))
    assert(Locus.parseInterval("X:START-101", rg) == Interval(Locus("X", 1), Locus("X", 101), true, false))
    assert(Locus.parseInterval("X:START-Y:END", rg) == Interval(Locus("X", 1), Locus("Y", yMax), true, true))
    assert(Locus.parseInterval("X-Y", rg) == Interval(Locus("X", 1), Locus("Y", yMax), true, true))
    assert(Locus.parseInterval("1-22", rg) == Interval(Locus("1", 1), Locus("22", chr22Max), true, true))

    // test normalizing end points
    assert(Locus.parseInterval(s"(X:100-${ xMax + 1 })", rg) == Interval(Locus("X", 100), Locus("X", xMax), false, true))
    assert(Locus.parseInterval(s"(X:0-$xMax]", rg) == Interval(Locus("X", 1), Locus("X", xMax), true, true))
    TestUtils.interceptFatal("Start 'X:0' is not within the range")(Locus.parseInterval("[X:0-5)", rg))
    TestUtils.interceptFatal(s"End 'X:${ xMax + 1 }' is not within the range")(Locus.parseInterval(s"[X:1-${ xMax + 1 }]", rg))

    assert(Locus.parseInterval("[16:29500000-30200000)", rg) == Interval(Locus("16", 29500000), Locus("16", 30200000), true, false))
    assert(Locus.parseInterval("[16:29.5M-30.2M)", rg) == Interval(Locus("16", 29500000), Locus("16", 30200000), true, false))
    assert(Locus.parseInterval("[16:29500K-30200K)", rg) == Interval(Locus("16", 29500000), Locus("16", 30200000), true, false))
    assert(Locus.parseInterval("[1:100K-2:200K)", rg) == Interval(Locus("1", 100000), Locus("2", 200000), true, false))

    assert(Locus.parseInterval("[1:1.111K-2000)", rg) == Interval(Locus("1", 1111), Locus("1", 2000), true, false))
    assert(Locus.parseInterval("[1:1.111111M-2000000)", rg) == Interval(Locus("1", 1111111), Locus("1", 2000000), true, false))

    TestUtils.interceptFatal("invalid interval expression") {
      Locus.parseInterval("4::start-5:end", rg)
    }

    TestUtils.interceptFatal("invalid interval expression") {
      Locus.parseInterval("4:start-", rg)
    }

    TestUtils.interceptFatal("invalid interval expression") {
      Locus.parseInterval("1:1.1111K-2k", rg)
    }

    TestUtils.interceptFatal("invalid interval expression") {
      Locus.parseInterval("1:1.1111111M-2M", rg)
    }

    val gr37 = ctx.getReference(ReferenceGenome.GRCh37)
    val gr38 = ctx.getReference(ReferenceGenome.GRCh38)

    val x = "[GL000197.1:3739-GL000202.1:7538)"
    assert(Locus.parseInterval(x, gr37) ==
      Interval(Locus("GL000197.1", 3739), Locus("GL000202.1", 7538), true, false))

    val y = "[HLA-DRB1*13:02:01:5-HLA-DRB1*14:05:01:100)"
    assert(Locus.parseInterval(y, gr38) ==
      Interval(Locus("HLA-DRB1*13:02:01", 5), Locus("HLA-DRB1*14:05:01", 100), true, false))

    val z = "[HLA-DRB1*13:02:01:5-100)"
    assert(Locus.parseInterval(z, gr38) ==
      Interval(Locus("HLA-DRB1*13:02:01", 5), Locus("HLA-DRB1*13:02:01", 100), true, false))
  }
}
