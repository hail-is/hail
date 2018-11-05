package is.hail.cxx

import is.hail.annotations.{Region, RegionValue}
import is.hail.expr.types.physical.PType
import is.hail.nativecode._
import is.hail.rvd.RVDContext

class RegionValueIterator(it: Iterator[RegionValue]) extends Iterator[Long] {

  def next(): Long = it.next().offset

  def hasNext: Boolean = it.hasNext
}

object CXXRegionValueIterator {
  def apply(itClass: TranslationUnitBuilder => Class): (Region, AnyRef) => CXXRegionValueIterator = {
    val tub = new TranslationUnitBuilder()
    val cls = itClass(tub)
    tub += cls
    tub.include("hail/ObjectArray.h")

    val st = Variable("st", "NativeStatus *")
    val region = Variable("region", "long")
    val ptr = Variable("it", "long")
    tub += new Function("NativeObjPtr", "make_iterator", Array(st, ptr),
      s"return std::make_shared<$cls>(reinterpret_cast<ObjectArray*>($ptr)->at(0));")

    tub += new Function("long", "next", Array(st, ptr, region),
      s"return reinterpret_cast<long>(reinterpret_cast<${ cls.name } *>($ptr)->next($st, reinterpret_cast<Region*>($region)));")
    tub += new Function("long", "has_next", Array(st, ptr, region),
      s"return reinterpret_cast<${ cls.name } *>($ptr)->has_next($st, reinterpret_cast<Region*>($region));")

    val mod = tub.result().build("-O1 -llz4")
    val key = mod.getKey
    val bin = mod.getBinary

    { case (r, v) => new CXXRegionValueIterator(key, bin, v, r) }
  }

}

class CXXRegionValueIterator(key: String, bin: Array[Byte], obj: AnyRef, region: Region) extends Iterator[RegionValue] with AutoCloseable {
  private[this] val st = new NativeStatus()
  private[this] val mod = new NativeModule(key, bin)
  private[this] val objArray = new ObjectArray(obj)

  private[this] val itF = mod.findPtrFuncL1(st, "make_iterator")
  assert(st.ok, st.toString)
  private[this] val nextF = mod.findLongFuncL2(st, "next")
  assert(st.ok, st.toString)
  private[this] val hasNextF = mod.findLongFuncL2(st, "has_next")
  assert(st.ok, st.toString)

  private[this] val ptr = new NativePtr(itF, st, objArray.get())
  itF.close()
  objArray.close()

  private[this] var has_next_called = false
  private[this] var has_next = false

  private[this] var next_called = false
  private[this] var nextRV: RegionValue = _

  def next(): RegionValue = {
    if (!next_called) {
      nextRV = RegionValue(region, nextF(st, ptr.get(), region.get()))
      has_next_called = false
      next_called = true
    }
    nextRV
  }

  def hasNext(): Boolean = {
    if (!has_next_called) {
      has_next = hasNextF(st, ptr.get(), region.get()) != 0
      has_next_called = true
      next_called = false
    }
    has_next
  }

  def close(): Unit = {
    hasNextF.close()
    nextF.close()
    ptr.close()
    mod.close()
    st.close()
  }

}