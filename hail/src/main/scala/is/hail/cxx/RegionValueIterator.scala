package is.hail.cxx

import is.hail.annotations.{Region, RegionValue}
import is.hail.expr.types.physical.PType
import is.hail.nativecode._
import is.hail.rvd.RVDContext
import is.hail.utils.{StagingIterator, StateMachine}

class RegionValueIterator(it: Iterator[RegionValue]) extends Iterator[Long] {

  def next(): Long = it.next().offset

  def hasNext: Boolean = it.hasNext
}

object CXXRegionValueIterator {
  def apply(itClass: TranslationUnitBuilder => Class): (Region, AnyRef) => StagingIterator[RegionValue] = {
    val tub = new TranslationUnitBuilder()
    val cls = itClass(tub)
    tub += cls
    tub.include("hail/ObjectArray.h")

    val st = Variable("st", "NativeStatus *")
    val region = Variable("region", "long")
    val ptr = Variable("it", "long")
    tub += new Function("NativeObjPtr", "make_iterator", Array(st, ptr, region),
      s"return std::make_shared<$cls>(reinterpret_cast<ObjectArray*>($ptr)->at(0), reinterpret_cast<Region*>($region), $st);")

    tub += new Function("long", "get", Array(st, ptr),
      s"""
         |${ cls.name } it = *reinterpret_cast<${ cls.name } *>($ptr);
         |return reinterpret_cast<long>(*it);
       """.stripMargin)
    tub += new Function("long", "advance", Array(st, ptr),
      s"""
         |auto it = ++(*reinterpret_cast<${ cls.name } *>($ptr));
         |return (*it == nullptr) ? 0 : 1;
       """.stripMargin)

    val mod = tub.result().build("-O1 -llz4")
    val key = mod.getKey
    val bin = mod.getBinary

    { case (r, v) => StagingIterator(new CXXStateMachine(key, bin, v, r)) }
  }

}

class CXXStateMachine(key: String, bin: Array[Byte], obj: AnyRef, region: Region) extends StateMachine[RegionValue] with AutoCloseable {
  private[this] val st = new NativeStatus()
  private[this] val mod = new NativeModule(key, bin)
  private[this] val objArray = new ObjectArray(obj)

  private[this] val itF = mod.findPtrFuncL2(st, "make_iterator")
  assert(st.ok, st.toString)
  private[this] val getF = mod.findLongFuncL1(st, "get")
  assert(st.ok, st.toString)
  private[this] val advanceF = mod.findLongFuncL1(st, "advance")
  assert(st.ok, st.toString)

  private[this] val ptr = new NativePtr(itF, st, objArray.get(), region.get())
  itF.close()
  objArray.close()

  var isValid: Boolean = true
  var value: RegionValue = _
  advance()

  def advance(): Unit = {
    isValid = advanceF(st, ptr.get()) != 0
    if (isValid) {
      value = RegionValue(region, getF(st, ptr.get()))
    } else {
      close()
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