package is.hail.expr.ir

import is.hail.expr.ir.defs._

import org.junit.jupiter.api.Test

class ExistsSuite {

  @Test def testContainsAggInStreamAggStream(): Unit = {
    val ir =
      ApplyAggOp(Collect())(0)
        .stream
        .streamAgg(_ => ApplyAggOp(Count())())

    assert(ContainsAgg(ir))
  }

  @Test def testContainsAggNotInStreamAggQuery(): Unit = {
    val ir = StreamRange(0, 10, 1).streamAgg(_ => ApplyAggOp(Count())())
    assert(!ContainsAgg(ir))
  }

  @Test def testContainsNonCommutativeAggInStreamAggStream(): Unit = {
    val ir =
      ApplyAggOp(Collect())(0)
        .stream
        .streamAgg(_ => ApplyAggOp(Count())())

    assert(ContainsNonCommutativeAgg(ir))
  }

  @Test def testContainsNonCommutativeAggNotInStreamAggQuery(): Unit = {
    val ir = StreamRange(0, 10, 1).streamAgg(ApplyAggOp(Collect())(_))
    assert(!ContainsNonCommutativeAgg(ir))
  }

  @Test def testContainsScanInStreamAggScanStream(): Unit = {
    val ir =
      ApplyScanOp(Collect())(0)
        .stream
        .streamAggScan(_ => ApplyScanOp(Count())())

    assert(ContainsScan(ir))
  }

  @Test def testContainsScanNotInStreamAggScanQuery(): Unit = {
    val ir = StreamRange(0, 10, 1).streamAggScan(_ => ApplyScanOp(Count())())
    assert(!ContainsScan(ir))
  }

  @Test def testScanAggFoldIsNotAggResult(): Unit = {
    val ir = aggFoldIR(I32(0), isScan = true)(_ + 1)(_ + _)
    assert(!IsAggResult(ir))
  }
}
