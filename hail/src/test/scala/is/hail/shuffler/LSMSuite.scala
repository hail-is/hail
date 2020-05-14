package is.hail.shuffler

import org.apache.log4j.Logger;
import is.hail.annotations._
import is.hail.expr.ir.ExecuteContext
import is.hail.expr.types.virtual._
import is.hail.expr.types.physical._
import is.hail.io.{ BufferSpec, TypedCodecSpec }
import is.hail.testUtils._
import is.hail.shuffler.ShuffleTestUtils._
import is.hail.utils._
import is.hail.{HailSuite, TestUtils}
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.{DataProvider, Test}

import scala.util.Random
import scala.language.implicitConversions

class LSMSuite extends HailSuite {
  val log = Logger.getLogger(this.getClass.getName());

  private[this] def assertStrictlyIncreasingPrefix(
    ord: UnsafeOrdering,
    values: Array[UnsafeRow],
    prefixLength: Int
  ): Unit = {
    if (!(prefixLength <= values.length)) {
      throw new AssertionError(s"$prefixLength <= ${values.length}")
    }

    if (values.length <= 1) {
      return
    }

    var prev = values(0)
    var i = 1
    while (i < prefixLength) {
      assert(ord.lt(prev.offset, values(i).offset),
        s"""values are not strictly increasing on [0, $prefixLength). We saw
           |${prev} and ${values(i)} at $i. Context: ${values.slice(i-3, i+3).toIndexedSeq}
           |""".stripMargin)
      prev = values(i)
      i += 1
    }
  }

  case class PartitionKeyParameters(
    nElements: Int,
    nPartitions: Int,
    description: String = ""
  )

  @DataProvider(name = "partitionKeyParameters")
  def partitionKeyParameters(): Array[PartitionKeyParameters] = {
    val small = 500
    assert(small < LSM.nKeySamples)
    Array(
      PartitionKeyParameters(10000, 0),
      PartitionKeyParameters(10000, 1),
      PartitionKeyParameters(10000, 100),
      PartitionKeyParameters(small, small + 50, "nPartitions > nElements"),
      PartitionKeyParameters(small, 5, "fewer keys than LSM nKeySamples"),
      PartitionKeyParameters(small, small, "1:1 nPartitions:nElements, small"),
      PartitionKeyParameters(10000, 10000, "1:1 nPartitions:nElements"),
      PartitionKeyParameters(1000000, 100, "big test")
    )
  }

  @Test(dataProvider = "partitionKeyParameters")
  def testPartitionKeys(params: PartitionKeyParameters) {
    val nElements = params.nElements
    val nPartitions = params.nPartitions
    ExecuteContext.scoped() { (ctx: ExecuteContext) =>
      val rowPType = structIntStringPType
      val rowType = rowPType.virtualType
      val key = Array("x")
      val codecs = new KeyedCodecSpec(
        ctx,
        rowType,
        TypedCodecSpec(rowPType, BufferSpec.unblockedUncompressed),
        key)
      using(new LSM(ctx.createTmpPath("lsm"), codecs)) { lsm =>
        val shuffled = Random.shuffle((0 until nElements).toIndexedSeq).toArray

        val values = new Array[Long](nElements)
        var i = 0
        while (i < nElements) {
          val key = struct(shuffled(i))
          val value = struct(shuffled(i), s"${shuffled(i)}")
          lsm.put(key, value)
          values(i) = value
          i += 1
        }

        assert(lsm.size == nElements)

        val partitionKeys = arrayOfUnsafeRow(
          codecs.decodedKeyPType,
          lsm.partitionKeys(nPartitions))
        val keyOrd = codecs.decodedKeyPType.unsafeOrdering

        if (nPartitions == 0) {
          assert(partitionKeys.length == 0)
        } else if (nPartitions < nElements) {
          assert(partitionKeys.length == nPartitions + 1)
          assertStrictlyIncreasingPrefix(
            keyOrd, partitionKeys, partitionKeys.length)
        } else {
          assert(partitionKeys.length == nPartitions + 1)
          assertStrictlyIncreasingPrefix(
            keyOrd, partitionKeys, nElements)

          val lastKey = partitionKeys(nElements - 1)

          var i = nElements
          while (i < partitionKeys.length) {
            if (!(keyOrd.compare(lastKey.offset, partitionKeys(i).offset) == 0)) {
              throw new AssertionError(
                s"""at position $i: ${lastKey.offset} should equal ${partitionKeys(i)}
                 |because nPartitions, $nPartitions, >= nElements $nElements. The extra
                 |partitions should all be upper and lower bounded by the same key""".stripMargin)
            }
            i += 1
          }
        }
      }
    }
  }
}
