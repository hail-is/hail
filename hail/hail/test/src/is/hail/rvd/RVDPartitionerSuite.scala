package is.hail.rvd

import is.hail.annotations.RowSeq
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.types.virtual.{TInt32, TStruct}
import is.hail.utils.Interval

import org.junit.jupiter.api.{BeforeEach, Test}

class RVDPartitionerSuite {
  val kType = TStruct(("A", TInt32), ("B", TInt32), ("C", TInt32))

  var partitioner: RVDPartitioner = _

  @BeforeEach
  def setupPartitioner(implicit ctx: ExecuteContext): Unit = {
    partitioner = new RVDPartitioner(
      ctx.stateManager,
      kType,
      ArraySeq(
        Interval(RowSeq(1, 0), RowSeq(4, 3), true, false),
        Interval(RowSeq(4, 3), RowSeq(7, 9), true, false),
        Interval(RowSeq(7, 11), RowSeq(10, 0), true, true),
      ),
    )
  }

  @Test def testExtendKey(implicit ctx: ExecuteContext): Unit = {
    val p = new RVDPartitioner(
      ctx.stateManager,
      TStruct(("A", TInt32), ("B", TInt32)),
      ArraySeq(
        Interval(RowSeq(1, 0), RowSeq(4, 3), true, true),
        Interval(RowSeq(4, 3), RowSeq(4, 3), true, true),
        Interval(RowSeq(4, 3), RowSeq(7, 9), true, false),
        Interval(RowSeq(7, 11), RowSeq(10, 0), true, true),
      ),
    )
    val extended = p.extendKey(kType)
    assert(extended.rangeBounds sameElements Array(
      Interval(RowSeq(1, 0), RowSeq(4, 3), true, true),
      Interval(RowSeq(4, 3), RowSeq(7, 9), false, false),
      Interval(RowSeq(7, 11), RowSeq(10, 0), true, true),
    ))
  }

  @Test def testGetPartitionWithPartitionKeys(): Unit = {
    assert(partitioner.lowerBound(RowSeq(-1, 7)) == 0)
    assert(partitioner.upperBound(RowSeq(-1, 7)) == 0)

    assert(partitioner.lowerBound(RowSeq(4, 2)) == 0)
    assert(partitioner.upperBound(RowSeq(4, 2)) == 1)

    assert(partitioner.lowerBound(RowSeq(4, 3)) == 1)
    assert(partitioner.upperBound(RowSeq(4, 3)) == 2)

    assert(partitioner.lowerBound(RowSeq(5, -10259)) == 1)
    assert(partitioner.upperBound(RowSeq(5, -10259)) == 2)

    assert(partitioner.lowerBound(RowSeq(7, 9)) == 2)
    assert(partitioner.upperBound(RowSeq(7, 9)) == 2)

    assert(partitioner.lowerBound(RowSeq(12, 19)) == 3)
    assert(partitioner.upperBound(RowSeq(12, 19)) == 3)
  }

  @Test def testGetPartitionWithLargerKeys(): Unit = {
    assert(partitioner.lowerBound(RowSeq(0, 1, 3)) == 0)
    assert(partitioner.upperBound(RowSeq(0, 1, 3)) == 0)

    assert(partitioner.lowerBound(RowSeq(2, 7, 5)) == 0)
    assert(partitioner.upperBound(RowSeq(2, 7, 5)) == 1)

    assert(partitioner.lowerBound(RowSeq(4, 2, 1, 2.7, "bar")) == 0)

    assert(partitioner.lowerBound(RowSeq(7, 9, 7)) == 2)
    assert(partitioner.upperBound(RowSeq(7, 9, 7)) == 2)

    assert(partitioner.lowerBound(RowSeq(11, 1, 42)) == 3)
  }

  @Test def testGetPartitionPKWithSmallerKeys(): Unit = {
    assert(partitioner.lowerBound(RowSeq(2)) == 0)
    assert(partitioner.upperBound(RowSeq(2)) == 1)

    assert(partitioner.lowerBound(RowSeq(4)) == 0)
    assert(partitioner.upperBound(RowSeq(4)) == 2)

    assert(partitioner.lowerBound(RowSeq(11)) == 3)
    assert(partitioner.upperBound(RowSeq(11)) == 3)
  }

  @Test def testGetPartitionRange(): Unit = {
    assert(partitioner.queryInterval(Interval(RowSeq(3, 4), RowSeq(7, 11), true, true)) == Seq(
      0,
      1,
      2,
    ))
    assert(partitioner.queryInterval(Interval(RowSeq(3, 4), RowSeq(7, 9), true, false)) == Seq(
      0,
      1,
    ))
    assert(partitioner.queryInterval(Interval(RowSeq(4), RowSeq(5), true, true)) == Seq(0, 1))
    assert(partitioner.queryInterval(Interval(RowSeq(4), RowSeq(5), false, true)) == Seq(1))
    assert(partitioner.queryInterval(Interval(RowSeq(-1, 7), RowSeq(0, 9), true, false)) == Seq())
  }

  @Test def testGetSafePartitionKeyRange(): Unit = {
    assert(partitioner.queryKey(RowSeq(0, 0)).isEmpty)
    assert(partitioner.queryKey(RowSeq(7, 10)).isEmpty)
    assert(partitioner.queryKey(RowSeq(7, 11)) == Range.inclusive(2, 2))
  }

  @Test def testGenerateDisjoint(implicit ctx: ExecuteContext): Unit = {
    val intervals = ArraySeq(
      Interval(RowSeq(1, 0, 4), RowSeq(4, 3, 2), true, false),
      Interval(RowSeq(4, 3, 5), RowSeq(7, 9, 1), true, false),
      Interval(RowSeq(7, 11, 3), RowSeq(10, 0, 1), true, true),
      Interval(RowSeq(11, 0, 2), RowSeq(11, 0, 15), false, true),
      Interval(RowSeq(11, 0, 15), RowSeq(11, 0, 20), true, false),
    )

    val p3 = RVDPartitioner.generate(ctx.stateManager, ArraySeq("A", "B", "C"), kType, intervals)
    assert(p3.satisfiesAllowedOverlap(2))
    assert(p3.rangeBounds sameElements
      Array(
        Interval(RowSeq(1, 0, 4), RowSeq(4, 3, 2), true, false),
        Interval(RowSeq(4, 3, 5), RowSeq(7, 9, 1), true, false),
        Interval(RowSeq(7, 11, 3), RowSeq(10, 0, 1), true, true),
        Interval(RowSeq(11, 0, 2), RowSeq(11, 0, 15), false, true),
        Interval(RowSeq(11, 0, 15), RowSeq(11, 0, 20), false, false),
      ))

    val p2 = RVDPartitioner.generate(ctx.stateManager, ArraySeq("A", "B"), kType, intervals)
    assert(p2.satisfiesAllowedOverlap(1))
    assert(p2.rangeBounds sameElements
      Array(
        Interval(RowSeq(1, 0, 4), RowSeq(4, 3), true, true),
        Interval(RowSeq(4, 3), RowSeq(7, 9, 1), false, false),
        Interval(RowSeq(7, 11, 3), RowSeq(10, 0, 1), true, true),
        Interval(RowSeq(11, 0, 2), RowSeq(11, 0, 20), false, false),
      ))

    val p1 = RVDPartitioner.generate(ctx.stateManager, ArraySeq("A"), kType, intervals)
    assert(p1.satisfiesAllowedOverlap(0))
    assert(p1.rangeBounds sameElements
      Array(
        Interval(RowSeq(1, 0, 4), RowSeq(4), true, true),
        Interval(RowSeq(4), RowSeq(7), false, true),
        Interval(RowSeq(7), RowSeq(10, 0, 1), false, true),
        Interval(RowSeq(11, 0, 2), RowSeq(11, 0, 20), false, false),
      ))
  }

  @Test def testGenerateEmptyKey(implicit ctx: ExecuteContext): Unit = {
    val intervals1 = ArraySeq(Interval(RowSeq(), RowSeq(), true, true))
    val intervals5 = ArraySeq.fill(5)(Interval(RowSeq(), RowSeq(), true, true))

    val p5 = RVDPartitioner.generate(ctx.stateManager, FastSeq(), TStruct.empty, intervals5)
    assert(p5.rangeBounds == intervals1)

    val p1 = RVDPartitioner.generate(ctx.stateManager, FastSeq(), TStruct.empty, intervals1)
    assert(p1.rangeBounds == intervals1)

    val p0 = RVDPartitioner.generate(ctx.stateManager, FastSeq(), TStruct.empty, FastSeq())
    assert(p0.rangeBounds.isEmpty)
  }

  @Test def testIntersect(implicit ctx: ExecuteContext): Unit = {
    val kType = TStruct(("key", TInt32))
    val left =
      new RVDPartitioner(
        ctx.stateManager,
        kType,
        ArraySeq(
          Interval(RowSeq(1), RowSeq(10), true, false),
          Interval(RowSeq(12), RowSeq(13), true, false),
          Interval(RowSeq(14), RowSeq(19), true, false),
        ),
      )
    val right =
      new RVDPartitioner(
        ctx.stateManager,
        kType,
        ArraySeq(
          Interval(RowSeq(1), RowSeq(4), true, false),
          Interval(RowSeq(4), RowSeq(5), true, false),
          Interval(RowSeq(7), RowSeq(16), true, true),
          Interval(RowSeq(19), RowSeq(20), true, true),
        ),
      )
    assert(left.intersect(right).rangeBounds sameElements
      Array(
        Interval(RowSeq(1), RowSeq(4), true, false),
        Interval(RowSeq(4), RowSeq(5), true, false),
        Interval(RowSeq(7), RowSeq(10), true, false),
        Interval(RowSeq(12), RowSeq(13), true, false),
        Interval(RowSeq(14), RowSeq(16), true, true),
      ))
  }
}
