package is.hail.expr.ir

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import is.hail.HailSuite
import is.hail.annotations.{CodeOrdering, Region, RegionUtils}
import is.hail.asm4s._
import is.hail.check.{Gen, Prop}
import is.hail.expr.ir.agg._
import is.hail.expr.types.physical._
import is.hail.io.{CodecSpec, InputBuffer, OutputBuffer, StreamBufferSpec}
import is.hail.utils._
import org.testng.annotations.Test

import scala.collection.mutable
class TestBTreeKey(mb: EmitMethodBuilder) extends BTreeKey {
  private val comp = mb.getCodeOrdering[Int](PInt64(), CodeOrdering.compare)
  def storageType: PTuple = PTuple(required = true, PInt64(), PTuple())
  def compType: PType = PInt64()
  def isEmpty(off: Code[Long]): Code[Boolean] =
    storageType.isFieldMissing(off, 1)
  def initializeEmpty(off: Code[Long]): Code[Unit] =
    storageType.setFieldMissing(off, 1)

  def storeKey(off: Code[Long], m: Code[Boolean], v: Code[Long]): Code[Unit] =
    Code(
      storageType.clearMissingBits(off),
      m.mux(
        storageType.setFieldMissing(off, 0),
        Region.storeLong(storageType.fieldOffset(off, 0), v)))

  def copy(src: Code[Long], dest: Code[Long]): Code[Unit] =
    Region.copyFrom(src, dest, storageType.byteSize)
  def deepCopy(er: EmitRegion, src: Code[Long], dest: Code[Long]): Code[Unit] =
    copy(src, dest)

  def compKeys(k1: (Code[Boolean], Code[_]), k2: (Code[Boolean], Code[_])): Code[Int] =
    comp(k1, k2)

  def loadCompKey(off: Code[Long]): (Code[Boolean], Code[_]) =
    storageType.isFieldMissing(off, 0) -> storageType.isFieldMissing(off, 0).mux(
      0L, Region.loadLong(storageType.fieldOffset(off, 0)))
}

object BTreeBackedSet {
  def bulkLoad(region: Region, serialized: Array[Byte]): BTreeBackedSet = {
    val fb = EmitFunctionBuilder[Region, InputBuffer, Long]
    val root = fb.newField[Long]
    val r = fb.newField[Region]
    val ib = fb.getArg[InputBuffer](2)
    val ib2 = fb.newField[InputBuffer]

    val km = fb.newField[Boolean]
    val kv = fb.newField[Long]

    val key = new TestBTreeKey(fb.apply_method)
    val btree = new AppendOnlyBTree(fb: EmitFunctionBuilder[_], key, r, root)
    fb.emit(Code(
      r := fb.getArg[Region](1),
      btree.init,
      ib2 := ib,
      btree.bulkLoad(ib2) { (ib, off) =>
        Code(
          km := ib.readBoolean(),
          kv := km.mux(0L, ib.readLong()),
          key.storeKey(off, km, kv))
      },
      root
    ))

    val inputBuffer = new StreamBufferSpec().buildInputBuffer(new ByteArrayInputStream(serialized))
    val set = new BTreeBackedSet(region)
    set.root = fb.resultWithIndex()(0, region)(region, inputBuffer)
    set
  }
}

class BTreeBackedSet(region: Region) {

  var root: Long = 0

  private val newTreeF = {
    val fb = EmitFunctionBuilder[Region, Long]
    val root = fb.newField[Long]
    val r = fb.newField[Region]

    val key = new TestBTreeKey(fb.apply_method)
    val btree = new AppendOnlyBTree(fb: EmitFunctionBuilder[_], key, r, root)
    fb.emit(Code(
      r := fb.getArg[Region](1),
      btree.init, root))

    fb.resultWithIndex()(0, region)
  }

  private val getF = {
    val fb = EmitFunctionBuilder[Region, Long, Boolean, Long, Long]
    val root = fb.newField[Long]
    val r = fb.newField[Region]
    val m = fb.getArg[Boolean](3)
    val v = fb.getArg[Long](4)
    val elt = fb.newLocal[Long]

    val key = new TestBTreeKey(fb.apply_method)
    val btree = new AppendOnlyBTree(fb: EmitFunctionBuilder[_], key, r, root)

    fb.emit(Code(
      r := fb.getArg[Region](1),
      root := fb.getArg[Long](2),
      elt := btree.getOrElseInitialize(m, v),
      key.isEmpty(elt).orEmpty(key.storeKey(elt, m, v)),
      root))
    fb.resultWithIndex()(0, region)
  }

  private val getResultsF = {
    val fb = EmitFunctionBuilder[Region, Long, Array[java.lang.Long]]
    val root = fb.newField[Long]
    val r = fb.newField[Region]

    val key = new TestBTreeKey(fb.apply_method)
    val btree = new AppendOnlyBTree(fb: EmitFunctionBuilder[_], key, r, root)

    val sab = new StagedArrayBuilder(PInt64(), fb.apply_method, 16)
    val idx = fb.newLocal[Int]
    val returnArray = fb.newLocal[Array[java.lang.Long]]

    fb.emit(Code(
      r := fb.getArg[Region](1),
      root := fb.getArg[Long](2),
      sab.clear,
      btree.foreach { koff =>
        val (m, v) = key.loadCompKey(koff)
        m.mux(sab.addMissing(),
            sab.add(v))
      },
      returnArray := Code.newArray[java.lang.Long](sab.size),
      idx := 0,
      Code.whileLoop(idx < sab.size,
        returnArray.update(idx, sab.isMissing(idx).mux(
          Code._null,
          Code.boxLong(coerce[Long](sab(idx))))),
        idx := idx + 1
      ),
      returnArray))
    fb.resultWithIndex()(0, region)
  }

  private val bulkStoreF = {
    val fb = EmitFunctionBuilder[Long, OutputBuffer, Unit]
    val root = fb.newField[Long]
    val r = fb.newField[Region]
    val ob = fb.getArg[OutputBuffer](2)
    val ob2 = fb.newField[OutputBuffer]

    val key = new TestBTreeKey(fb.apply_method)
    val btree = new AppendOnlyBTree(fb: EmitFunctionBuilder[_], key, r, root)

    fb.emit(Code(
      root := fb.getArg[Long](1),
      ob2 := ob,
      btree.bulkStore(ob2) { (ob, off) =>
        val (km, kv) = key.loadCompKey(off)
        Code(
          ob.writeBoolean(km),
          (!km).orEmpty(ob.writeLong(coerce[Long](kv))))
      },
      ob2.load().flush()))

    fb.resultWithIndex()(0, region)
  }

  def clear(): Unit = {
    if (root != 0) { region.clear() }
    root = newTreeF(region)
  }

  def getOrElseInsert(v: java.lang.Long): Unit =
    root = getF(region, root, v == null, if (v == null) 0L else v.longValue())

  def getElements: Array[java.lang.Long] =
    getResultsF(region, root)

  def bulkStore: Array[Byte] = {
    val baos = new ByteArrayOutputStream()
    val outputBuffer = new StreamBufferSpec().buildOutputBuffer(baos)
    bulkStoreF(root, outputBuffer)
    baos.toByteArray
  }
}

class TestSet {
  val map = mutable.Set[java.lang.Long]()

  def clear(): Unit = map.clear()

  def getOrElseInsert(v: java.lang.Long): Unit = map += v

  def getElements: Array[java.lang.Long] = map.toArray
}

class StagedBTreeSuite extends HailSuite {

  @Test def testBTree(): Unit = {
    Region.scoped { region =>
      val refSet = new TestSet()
      val testSet = new BTreeBackedSet(region)

      val sets = Gen.buildableOf[Array](Gen.zip(Gen.coin(.1), Gen.choose(-10, 10))
        .map { case (m, v) => if (m) null else new java.lang.Long(v) })
      val lt = { (l1: java.lang.Long, l2: java.lang.Long) =>
        !(l1 == null) && ((l2 == null) || (l1 < l2))
      }

      Prop.forAll(sets) { set =>
        refSet.clear()
        testSet.clear()
        assert(refSet.getElements sameElements testSet.getElements)

        set.forall { v =>
          refSet.getOrElseInsert(v)
          testSet.getOrElseInsert(v)

          refSet.getElements.sortWith(lt) sameElements testSet.getElements.sortWith(lt)
        } && {
          val serialized = testSet.bulkStore
          val testSet2 = BTreeBackedSet.bulkLoad(region, serialized)
          refSet.getElements.sortWith(lt) sameElements testSet2.getElements.sortWith(lt)
        }
      }.check()
    }
  }
}
