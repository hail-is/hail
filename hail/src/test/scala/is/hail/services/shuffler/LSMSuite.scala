package is.hail.services.shuffler

import org.apache.log4j.Logger;
import is.hail.annotations._
import is.hail.expr.ir._
import is.hail.types.virtual._
import is.hail.types.physical._
import is.hail.types.encoded._
import is.hail.services.shuffler.server._
import is.hail.services.shuffler.ShufflerTestUtils._
import is.hail.io.{ BufferSpec, TypedCodecSpec }
import is.hail.testUtils._
import is.hail.utils._
import is.hail.{HailSuite, TestUtils}
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.{DataProvider, Test}

import scala.util.Random
import scala.language.implicitConversions

class LSMSuite extends HailSuite {
  val log = Logger.getLogger(this.getClass.getName());

  def testPartitionKeysFiveElementsThreePartitions() {
    val nElements = 5
    val nPartitions = 3
    ExecuteContext.scoped() { (ctx: ExecuteContext) =>
      val rowPType = structIntStringPType
      val rowEType = EType.defaultFromPType(rowPType).asInstanceOf[EBaseStruct]
      val rowType = rowPType.virtualType
      val key = Array(SortField("x", Ascending))
      val keyType = rowType.typeAfterSelectNames(key.map(_.field))
      val keyPType = PType.canonical(keyType)
      val keyEType = EType.defaultFromPType(keyPType).asInstanceOf[EBaseStruct]
      val codecs = new ShuffleCodecSpec(ctx, TShuffle(key, rowType, rowEType, keyEType))
      using(new LSM(ctx.createTmpPath("lsm"), codecs)) { lsm =>
        val shuffled = Array(4, 3, 1, 2, 0)

        lsm.put(struct(4), struct(4, "4"))
        lsm.put(struct(3), struct(3, "3"))
        lsm.put(struct(1), struct(1, "1"))
        lsm.put(struct(2), struct(2, "2"))
        lsm.put(struct(0), struct(0, "0"))

        assert(lsm.size == nElements)

        val partitionKeys = arrayOfUnsafeRow(
          codecs.keyDecodedPType,
          lsm.partitionKeys(nPartitions))
        val keyOrd = codecs.keyDecodedPType.unsafeOrdering

        assert(partitionKeys.length == nPartitions + 1)

        assert(partitionKeys(0).getInt(0) == 0)
        assert(partitionKeys(0).getInt(0) == 2)
        assert(partitionKeys(0).getInt(0) == 4)
        assert(partitionKeys(0).getInt(0) == 5)
      }
    }
  }

  def testPartitionKeysFiveElementsSevenPartitions() {
    val nElements = 5
    val nPartitions = 7
    ExecuteContext.scoped() { (ctx: ExecuteContext) =>
      val rowPType = structIntStringPType
      val rowEType = EType.defaultFromPType(rowPType).asInstanceOf[EBaseStruct]
      val rowType = rowPType.virtualType
      val key = Array(SortField("x", Ascending))
      val keyType = rowType.typeAfterSelectNames(key.map(_.field))
      val keyPType = PType.canonical(keyType)
      val keyEType = EType.defaultFromPType(keyPType).asInstanceOf[EBaseStruct]
      val codecs = new ShuffleCodecSpec(ctx, TShuffle(key, rowType, rowEType, keyEType))
      using(new LSM(ctx.createTmpPath("lsm"), codecs)) { lsm =>
        val shuffled = Array(4, 3, 1, 2, 0)

        lsm.put(struct(4), struct(4, "4"))
        lsm.put(struct(3), struct(3, "3"))
        lsm.put(struct(1), struct(1, "1"))
        lsm.put(struct(2), struct(2, "2"))
        lsm.put(struct(0), struct(0, "0"))

        assert(lsm.size == nElements)

        val partitionKeys = arrayOfUnsafeRow(
          codecs.keyDecodedPType,
          lsm.partitionKeys(nPartitions))
        val keyOrd = codecs.keyDecodedPType.unsafeOrdering

        assert(partitionKeys.length == nElements + 1)

        assert(partitionKeys(0).getInt(0) == 0)
        assert(partitionKeys(0).getInt(0) == 1)
        assert(partitionKeys(0).getInt(0) == 2)
        assert(partitionKeys(0).getInt(0) == 3)
        assert(partitionKeys(0).getInt(0) == 4)
        assert(partitionKeys(0).getInt(0) == 4)
        assert(partitionKeys(0).getInt(0) == 4)
        assert(partitionKeys(0).getInt(0) == 4)
      }
    }
  }

  def testNoPartitions() {
    testPartitionKeys(nElements = 100000, nPartitions = 0)
  }

  def testOnePartitions() {
    testPartitionKeys(nElements = 100000, nPartitions = 1)
  }

  def testOneHundredPartitions() {
    testPartitionKeys(nElements = 100000, nPartitions = 100)
  }

  private[this] val fewerKeysThanSamples = 500
  assert(fewerKeysThanSamples < LSM.nKeySamples)

  def testFewerKeysThanSamplesMorePartitionsThanElements() {
    testPartitionKeys(fewerKeysThanSamples, fewerKeysThanSamples + 50)
  }

  def testFewerKeysThanSamplesFivePartitions() {
    testPartitionKeys(fewerKeysThanSamples, 5)
  }

  def testFewerKeysThanSamplesOneToOnePartitions() {
    testPartitionKeys(fewerKeysThanSamples, fewerKeysThanSamples)
  }

  def testLargerFewerKeysThanSamplesOneToOnePartitions() {
    testPartitionKeys(10000, 10000)
  }

  def testHuge() {
    testPartitionKeys(1000000, 100)
  }

  def testPartitionKeys(nElements: Int, nPartitions: Int) {
    ExecuteContext.scoped() { (ctx: ExecuteContext) =>
      val rowPType = structIntStringPType
      val rowEType = EType.defaultFromPType(rowPType).asInstanceOf[EBaseStruct]
      val rowType = rowPType.virtualType
      val key = Array(SortField("x", Ascending))
      val keyType = rowType.typeAfterSelectNames(key.map(_.field))
      val keyPType = PType.canonical(keyType)
      val keyEType = EType.defaultFromPType(keyPType).asInstanceOf[EBaseStruct]
      val codecs = new ShuffleCodecSpec(ctx, TShuffle(key, rowType, rowEType, keyEType))
      using(new LSM(ctx.createTmpPath("lsm"), codecs)) { lsm =>
        val shuffled = Random.shuffle((0 until nElements).toIndexedSeq).toArray

        var i = 0
        while (i < nElements) {
          val key = struct(shuffled(i))
          val value = struct(shuffled(i), s"${shuffled(i)}")
          lsm.put(key, value)
          i += 1
        }

        assert(lsm.size == nElements)

        val partitionKeys = arrayOfUnsafeRow(
          codecs.keyDecodedPType,
          lsm.partitionKeys(nPartitions))
        val keyOrd = codecs.keyDecodedPType.unsafeOrdering

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
