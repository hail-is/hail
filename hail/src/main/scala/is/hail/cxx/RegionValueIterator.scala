package is.hail.cxx

import is.hail.annotations.{Region, RegionValue}
import is.hail.nativecode._
import is.hail.utils.{FlipbookIterator, StagingIterator, StateMachine}

class RegionValueIterator(it: Iterator[RegionValue]) extends Iterator[Long] {

  def next(): Long = it.next().offset

  def hasNext: Boolean = it.hasNext
}

object CXXRegionValueIterator {
  def apply[T](itTyp: Type, tub: TranslationUnitBuilder, makeItF: (NativeModule, Region, T) => NativePtr): (Region, T) => FlipbookIterator[RegionValue] = {
    val st = tub.variable("st", "NativeStatus *")
    val ptr = tub.variable("it", "long")
    tub += new Function("long", "get", Array(st, ptr), s"return reinterpret_cast<long>(reinterpret_cast<${ itTyp } *>($ptr)->get());")
    tub += new Function("long", "advance", Array(st, ptr), s"return (reinterpret_cast<${ itTyp } *>($ptr)->advance()) ? 1 : 0;")

    val mod = tub.end().build("-O2")
    val key = mod.getKey
    val bin = mod.getBinary

    { case (r, v) => StagingIterator(new CXXStateMachine(key, bin, makeItF(_, r, v))).map(RegionValue(r, _)) }
  }
}

class CXXStateMachine(key: String, bin: Array[Byte], iteratorF: NativeModule => NativePtr) extends StateMachine[Long] with AutoCloseable {
  private[this] val st = new NativeStatus()
  private[this] val mod = new NativeModule(key, bin)
  private[this] val getF = mod.findLongFuncL1(st, "get")
  assert(st.ok, st.toString)
  private[this] val advanceF = mod.findLongFuncL1(st, "advance")
  assert(st.ok, st.toString)

  private[this] val ptr = iteratorF(mod)

  var isValid: Boolean = true
  var value: Long = getF(st, ptr.get())

  def advance(): Unit = {
    isValid = advanceF(st, ptr.get()) != 0
    if (isValid) {
      value = getF(st, ptr.get())
    }
  }

  def close(): Unit = {
    advanceF.close()
    getF.close()
    ptr.close()
    mod.close()
    st.close()
  }

}
