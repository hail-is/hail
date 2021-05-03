package is.hail.expr.ir

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.types.physical.{PCanonicalArray, PCanonicalDict, PCanonicalSet, PCode, PIndexableCode, PType, typeToTypeInfo}
import is.hail.types.virtual.{TArray, TDict, TSet, Type}
import is.hail.utils.FastIndexedSeq

class ArraySorter(r: EmitRegion, array: StagedArrayBuilder) {
  val ti: TypeInfo[_] = array.elt.ti
  val mb: EmitMethodBuilder[_] = r.mb

  def sort(cb: EmitCodeBuilder, region: Value[Region], comparesLessThan: (EmitCodeBuilder, Value[Region], Code[_], Code[_]) => Code[Boolean]): Unit = {

    val sortMB = cb.emb.ecb.newEmitMethod("arraySorter_outer", FastIndexedSeq[ParamType](classInfo[Region]), UnitInfo)
    sortMB.voidWithBuilder { cb =>

      val newEnd = cb.newLocal[Int]("newEnd", 0)
      val i = cb.newLocal[Int]("i", 0)
      val size = cb.newLocal[Int]("size", array.size)

      cb.whileLoop(i < size, {
        cb.ifx(!array.isMissing(i), {
          cb.ifx(newEnd.cne(i), cb += array.update(newEnd, array.apply(i)))
          cb.assign(newEnd, newEnd + 1)
        })
        cb.assign(i, i + 1)
      })
      cb.assign(i, newEnd)
      cb.whileLoop(i < size, {
        cb += array.setMissing(i, true)
        cb.assign(i, i + 1)
      })

      // sort elements in [0, newEnd]

      val quickSortMB = cb.emb.ecb.newEmitMethod("arraySorter_quicksort", FastIndexedSeq[ParamType](classInfo[Region], IntInfo, IntInfo), UnitInfo)
      quickSortMB.voidWithBuilder { cb =>
        val r = quickSortMB.getCodeParam[Region](1)
        val low = quickSortMB.getCodeParam[Int](2)
        val n = quickSortMB.getCodeParam[Int](3)

        def swap(i: Value[Int], j: Value[Int]) {
          val tmp = cb.newLocalAny("tmp", array(i))(ti)
          cb += array.update(i, array(j))
          cb += array.update(j, tmp)
        }

        cb.ifx(n > 1, {
          val pivotIdx = cb.newLocal[Int]("pivotIdx", low + (n / 2))
          val pivot = cb.newLocalAny("pivot", array(pivotIdx))(ti)

          val left = cb.newLocal[Int]("left", low)
          val right = cb.newLocal[Int]("right", low + n - 1)
          swap(pivotIdx, right)

          cb.whileLoop(left < right, {
            cb.ifx(!comparesLessThan(cb, r, array(left), pivot),
              cb.assign(left, left + 1),
              cb.ifx(comparesLessThan(cb, r, array(right - 1), pivot),
                cb.assign(right, right - 1),
                {
                  swap(left, cb.newLocal[Int]("rightMinusOne", right - 1))
                  cb.assign(left, left + 1)
                  cb.assign(right, right - 1)
                },
              ))
          })

          swap(left, cb.newLocal("newEnd", low + n - 1))

          cb.invokeVoid(quickSortMB, r, low, left - low)
          cb.invokeVoid(quickSortMB, r, left + 1, n - (left - low - 1))
        })
      }

      cb.invokeCode(quickSortMB, sortMB.getCodeParam[Region](1), const(0), newEnd)
    }
    cb.invokeVoid(sortMB, region)
  }

  def toRegion(cb: EmitCodeBuilder, t: Type): PIndexableCode = {
    t match {
      case pca: TArray =>
        val len = cb.newLocal[Int]("arraysorter_to_region_len", array.size)
        // fixme element requiredness should be set here
        val arrayType = PCanonicalArray(array.elt.loadedSType.canonicalPType())

        arrayType.constructFromElements(cb, r.region, len, deepCopy = false) { (cb, idx) =>
          array.loadFromIndex(cb, r.region, idx)
        }
      case td: TDict =>
        PCanonicalDict.coerceArrayCode(toRegion(cb, TArray(td.elementType)))
      case ts: TSet =>
        PCanonicalSet.coerceArrayCode(toRegion(cb, TArray(ts.elementType)))
    }
  }

  def pruneMissing(cb: EmitCodeBuilder): Unit = {
    val i = cb.newLocal[Int]("i", 0)
    val n = cb.newLocal[Int]("n", 0)
    val size = cb.newLocal[Int]("size", array.size)
    cb.whileLoop(i < size, {
      cb.ifx(!array.isMissing(i), {
        cb.ifx(i.cne(n),
          cb += array.update(n, array(i)))
        cb.assign(n, n + 1)
      })
      cb.assign(i, i + 1)
    })
    cb += array.setSize(n)
  }

  def distinctFromSorted(cb: EmitCodeBuilder, region: Value[Region], discardNext: (EmitCodeBuilder, Value[Region], EmitCode, EmitCode) => Code[Boolean]): Unit = {

    val distinctMB = cb.emb.newEmitMethod("distinctFromSorted", FastIndexedSeq[ParamType](classInfo[Region]), UnitInfo)
    distinctMB.voidWithBuilder { cb =>
      val region = distinctMB.getCodeParam[Region](1)
      val i = cb.newLocal[Int]("i", 0)
      val n = cb.newLocal[Int]("n", 0)
      val size = cb.newLocal[Int]("size", array.size)
      cb.whileLoop(i < size, {
        cb.assign(i, i + 1)

        val LskipLoopBegin = CodeLabel()
        val LskipLoopEnd = CodeLabel()
        cb.define(LskipLoopBegin)
        cb.ifx(i >= size, cb.goto(LskipLoopEnd))
        cb.ifx(!discardNext(cb, region,
          EmitCode.fromI(distinctMB)(cb => array.loadFromIndex(cb, region, n)),
          EmitCode.fromI(distinctMB)(cb => array.loadFromIndex(cb, region, i))),
          cb.goto(LskipLoopEnd))
        cb.assign(i, i + 1)
        cb.goto(LskipLoopBegin)

        cb.define(LskipLoopEnd)

        cb.assign(n, n + 1)
        cb.ifx(i < size && i.cne(n), {
          cb += array.setMissing(n, array.isMissing(i))
          cb.ifx(!array.isMissing(n), cb += array.update(n, array(i)))
        })

      })
      cb += array.setSize(n)
    }
    cb.invokeVoid(distinctMB, region)
  }
}
