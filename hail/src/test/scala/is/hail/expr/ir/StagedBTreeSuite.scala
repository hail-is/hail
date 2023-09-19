package is.hail.expr.ir

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}
import is.hail.HailSuite
import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.check.{Gen, Prop}
import is.hail.expr.ir.agg._
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.types.physical._
import is.hail.io.{InputBuffer, OutputBuffer, StreamBufferSpec}
import is.hail.types.physical.stypes.Int64SingleCodeType
import is.hail.types.physical.stypes.interfaces.primitive
import is.hail.types.physical.stypes.primitives.SInt64
import is.hail.utils._
import org.testng.annotations.Test

import scala.collection.mutable
class TestBTreeKey(mb: EmitMethodBuilder[_]) extends BTreeKey {
  private val comp = mb.ecb.getOrderingFunction(SInt64, SInt64, CodeOrdering.Compare())
  override def storageType: PTuple = PCanonicalTuple(required = true, PInt64(), PCanonicalTuple(false))
  override def compType: PType = PInt64()
  override def isEmpty(cb: EmitCodeBuilder, off: Code[Long]): Value[Boolean] =
    storageType.isFieldMissing(cb, off, 1)
  override def initializeEmpty(cb: EmitCodeBuilder, off: Code[Long]): Unit =
    storageType.setFieldMissing(cb, off, 1)

  def storeKey(cb: EmitCodeBuilder, _off: Code[Long], m: Code[Boolean], v: Code[Long]): Unit = {
    val off = cb.memoize[Long](_off)
    storageType.stagedInitialize(cb, off)
    cb.ifx(m,
      storageType.setFieldMissing(cb, off, 0),
      cb += Region.storeLong(storageType.fieldOffset(off, 0), v)
    )
  }

  override def copy(cb: EmitCodeBuilder, src: Code[Long], dest: Code[Long]): Unit =
    cb += Region.copyFrom(src, dest, storageType.byteSize)
  override def deepCopy(cb: EmitCodeBuilder, er: EmitRegion, src: Code[Long], dest: Code[Long]): Unit =
    copy(cb, src, dest)

  override def compKeys(cb: EmitCodeBuilder, k1: EmitValue, k2: EmitValue): Value[Int] =
    comp(cb, k1, k2)

  override def loadCompKey(cb: EmitCodeBuilder, off: Value[Long]): EmitValue =
    EmitValue(
      Some(storageType.isFieldMissing(cb, off, 0)),
      primitive(cb.memoize(Region.loadLong(storageType.fieldOffset(off, 0)))))
}

object BTreeBackedSet {
  def bulkLoad(ctx: ExecuteContext, region: Region, serialized: Array[Byte], n: Int): BTreeBackedSet = {
    val fb = EmitFunctionBuilder[Region, InputBuffer, Long](ctx, "btree_bulk_load")
    val cb = fb.ecb
    val root = fb.genFieldThisRef[Long]()
    val r = fb.genFieldThisRef[Region]()
    val ib = fb.getCodeParam[InputBuffer](2)
    val ib2 = fb.genFieldThisRef[InputBuffer]()

    val km = fb.genFieldThisRef[Boolean]()
    val kv = fb.genFieldThisRef[Long]()

    val key = new TestBTreeKey(fb.apply)
    val btree = new AppendOnlyBTree(cb, key, r, root, maxElements = n)
    fb.emitWithBuilder { cb =>
      cb += (r := fb.getCodeParam[Region](1))
      btree.init(cb)
      btree.bulkLoad(cb, ib) { (cb, ib, off) =>
        cb.assign(km, ib.readBoolean())
        cb.assign(kv, km.mux(0L, ib.readLong()))
        key.storeKey(cb, off, km, kv)
      }
      root
    }

    val inputBuffer = new StreamBufferSpec().buildInputBuffer(new ByteArrayInputStream(serialized))
    val set = new BTreeBackedSet(ctx, region, n)
    set.root = fb.resultWithIndex()(HailSuite.theHailClassLoader, ctx.fs, ctx.taskContext, region)(region, inputBuffer)
    set
  }
}

class BTreeBackedSet(ctx: ExecuteContext, region: Region, n: Int) {

  var root: Long = 0

  private val newTreeF = {
    val fb = EmitFunctionBuilder[Region, Long](ctx, "new_tree")
    val cb = fb.ecb
    val root = fb.genFieldThisRef[Long]()
    val r = fb.genFieldThisRef[Region]()

    val key = new TestBTreeKey(fb.apply)
    val btree = new AppendOnlyBTree(cb, key, r, root, maxElements = n)
    fb.emitWithBuilder { cb =>
      cb.assign(r, fb.getCodeParam[Region](1))
      btree.init(cb)
      root
    }

    fb.resultWithIndex()(HailSuite.theHailClassLoader, ctx.fs, ctx.taskContext, region)
  }

  private val getF = {
    val fb = EmitFunctionBuilder[Region, Long, Boolean, Long, Long](ctx, "get")
    val cb = fb.ecb
    val root = fb.genFieldThisRef[Long]()
    val r = fb.genFieldThisRef[Region]()
    val m = fb.getCodeParam[Boolean](3)
    val v = fb.getCodeParam[Long](4)
    val elt = fb.newLocal[Long]()

    val key = new TestBTreeKey(fb.apply)
    val btree = new AppendOnlyBTree(cb, key, r, root, maxElements = n)

    fb.emitWithBuilder { cb =>
      val ec = EmitCode(Code._empty, m, primitive(v))
      cb.assign(r, fb.getCodeParam[Region](1))
      cb.assign(root, fb.getCodeParam[Long](2))
      cb.assign(elt, btree.getOrElseInitialize(cb, ec))
      cb.ifx(key.isEmpty(cb, elt), {
        key.storeKey(cb, elt, m, v)
      })
      root
    }
    fb.resultWithIndex()(HailSuite.theHailClassLoader, ctx.fs, ctx.taskContext, region)
  }

  private val getResultsF = {
    val fb = EmitFunctionBuilder[Region, Long, Array[java.lang.Long]](ctx, "get_results")
    val cb = fb.ecb
    val root = fb.genFieldThisRef[Long]()
    val r = fb.genFieldThisRef[Region]()

    val key = new TestBTreeKey(fb.apply)
    val btree = new AppendOnlyBTree(cb, key, r, root, maxElements = n)

    val sab = new StagedArrayBuilder(Int64SingleCodeType, true, fb.apply, 16)
    val idx = fb.newLocal[Int]()
    val returnArray = fb.newLocal[Array[java.lang.Long]]()

    fb.emitWithBuilder { cb =>
      cb += (r := fb.getCodeParam[Region](1))
      cb += (root := fb.getCodeParam[Long](2))
      cb += sab.clear
      btree.foreach(cb) { (cb, _koff) =>
        val koff = cb.memoize(_koff)
        val ec = key.loadCompKey(cb, koff)
        cb.ifx(ec.m,
          cb += sab.addMissing(),
          cb += sab.add(ec.pv.asInt64.value))
      }
      cb += (returnArray := Code.newArray[java.lang.Long](sab.size))
      cb += (idx := 0)
      cb += Code.whileLoop(idx < sab.size,
        returnArray.update(idx, sab.isMissing(idx).mux(
          Code._null[java.lang.Long],
          Code.boxLong(coerce[Long](sab(idx))))),
        idx := idx + 1
      )
      returnArray
    }
    fb.resultWithIndex()(HailSuite.theHailClassLoader, ctx.fs, ctx.taskContext, region)
  }

  private val bulkStoreF = {
    val fb = EmitFunctionBuilder[Long, OutputBuffer, Unit](ctx, "bulk_store")
    val cb = fb.ecb
    val root = fb.genFieldThisRef[Long]()
    val r = fb.genFieldThisRef[Region]()
    val ob = fb.getCodeParam[OutputBuffer](2)
    val ob2 = fb.genFieldThisRef[OutputBuffer]()

    val key = new TestBTreeKey(fb.apply)
    val btree = new AppendOnlyBTree(cb, key, r, root, maxElements = n)

    fb.emitWithBuilder { cb =>
      cb += (root := fb.getCodeParam[Long](1))
      cb += (ob2 := ob)
      btree.bulkStore(cb, ob2) { (cb, obc, offc) =>
        val ob = cb.newLocal("ob", obc)
        val off = cb.newLocal("off", offc)
        val ev = cb.memoize(key.loadCompKey(cb, off), "ev")
        cb += ob.writeBoolean(ev.m)
        cb.ifx(!ev.m, {
          cb += ob.writeLong(ev.pv.asInt64.value)
        })
      }
      ob2.flush()
    }

    fb.resultWithIndex()(HailSuite.theHailClassLoader, ctx.fs, ctx.taskContext, region)
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
    pool.scopedRegion { region =>
      val refSet = new TestSet()
      val nodeSizeParams = Array(
        2 -> Gen.choose(-10, 10),
        3 -> Gen.choose(-10, 10),
        5 -> Gen.choose(-30, 30),
        6 -> Gen.choose(-30, 30),
        22 -> Gen.choose(-3, 3))

      for ((n, values) <- nodeSizeParams) {
        val testSet = new BTreeBackedSet(ctx, region, n)

        val sets = Gen.buildableOf[Array](Gen.zip(Gen.coin(.1), values)
          .map { case (m, v) => if (m) null else Long.box(v.longValue()) })
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
            val testSet2 = BTreeBackedSet.bulkLoad(ctx, region, serialized, n)
            refSet.getElements.sortWith(lt) sameElements testSet2.getElements.sortWith(lt)
          }
        }.check()
      }
    }
  }
}
