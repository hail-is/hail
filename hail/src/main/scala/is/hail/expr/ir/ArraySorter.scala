package is.hail.expr.ir

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.types.physical.stypes.interfaces.SIndexableCode
import is.hail.types.physical.{PCanonicalArray, PCanonicalDict, PCanonicalSet}
import is.hail.types.virtual.{TArray, TDict, TSet, Type}
import is.hail.utils.FastIndexedSeq

import scala.language.existentials

class ArraySorter(r: EmitRegion, array: StagedArrayBuilder) {
  val ti: TypeInfo[_] = array.elt.ti
  val mb: EmitMethodBuilder[_] = r.mb

  private[this] var prunedMissing: Boolean = false

  private[this] val workingArrayInfo = arrayInfo(array.ti)
  private[this] val workingArray1 = mb.genFieldThisRef("sorter_working_array")(workingArrayInfo)
  private[this] val workingArray2 = mb.genFieldThisRef("sorter_working_array")(workingArrayInfo)

  private[this] def arrayRef(workingArray: Code[Array[_]]): UntypedCodeArray = new UntypedCodeArray(workingArray, array.ti)

  def sort(cb: EmitCodeBuilder, region: Value[Region], comparesLessThan: (EmitCodeBuilder, Value[Region], Code[_], Code[_]) => Code[Boolean]): Unit = {

    val sortMB = cb.emb.ecb.genEmitMethod("arraySorter_outer", FastIndexedSeq[ParamType](classInfo[Region]), UnitInfo)
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

      // merging into B
      val mergeMB = cb.emb.ecb.genEmitMethod("arraySorter_merge", FastIndexedSeq[ParamType](classInfo[Region], IntInfo, IntInfo, IntInfo, workingArrayInfo, workingArrayInfo), UnitInfo)
      mergeMB.voidWithBuilder { cb =>
        val r = mergeMB.getCodeParam[Region](1)
        val begin = mergeMB.getCodeParam[Int](2)
        val mid = mergeMB.getCodeParam[Int](3)
        val end = mergeMB.getCodeParam[Int](4)

        def arrayA = new UntypedCodeArray(mergeMB.getCodeParam(5)(workingArrayInfo), array.ti)

        def arrayB = new UntypedCodeArray(mergeMB.getCodeParam(6)(workingArrayInfo), array.ti)

        val i = cb.newLocal[Int]("mergemb_i", begin)
        val j = cb.newLocal[Int]("mergemb_j", mid)

        val k = cb.newLocal[Int]("mergemb_k", i)
        cb.whileLoop(k < end, {

          val LtakeFromLeft = CodeLabel()
          val LtakeFromRight = CodeLabel()
          val Ldone = CodeLabel()

          cb.ifx(j < end, {
            cb.ifx(i >= mid, cb.goto(LtakeFromRight))
            cb.ifx(comparesLessThan(cb, r, arrayA(j), arrayA(i)), cb.goto(LtakeFromRight), cb.goto(LtakeFromLeft))
          }, cb.goto(LtakeFromLeft))

          cb.define(LtakeFromLeft)
          cb += arrayB.update(k, arrayA(i))
          cb.assign(i, i + 1)
          cb.goto(Ldone)

          cb.define(LtakeFromRight)
          cb += arrayB.update(k, arrayA(j))
          cb.assign(j, j + 1)
          cb.goto(Ldone)

          cb.define(Ldone)
          cb.assign(k, k + 1)
        })
      }

      val splitMergeMB = cb.emb.ecb.genEmitMethod("arraySorter_splitMerge", FastIndexedSeq[ParamType](classInfo[Region], IntInfo, IntInfo, workingArrayInfo, workingArrayInfo), UnitInfo)
      splitMergeMB.voidWithBuilder { cb =>
        val r = splitMergeMB.getCodeParam[Region](1)
        val begin = splitMergeMB.getCodeParam[Int](2)
        val end = splitMergeMB.getCodeParam[Int](3)

        val arrayB = splitMergeMB.getCodeParam(4)(workingArrayInfo)
        val arrayA = splitMergeMB.getCodeParam(5)(workingArrayInfo)

        cb.ifx(end - begin > 1, {
          val mid = cb.newLocal[Int]("splitMerge_mid", (begin + end) / 2)

          cb.invokeVoid(splitMergeMB, r, begin, mid, arrayA, arrayB)
          cb.invokeVoid(splitMergeMB, r, mid, end, arrayA, arrayB)

          // result goes in A
          cb.invokeVoid(mergeMB, r, begin, mid, end, arrayB, arrayA)
        })
      }

      // these arrays should be allocated once and reused
      cb.ifx(workingArray1.isNull || arrayRef(workingArray1).length() < newEnd, {
        cb.assignAny(workingArray1, Code.newArray(newEnd)(array.ti))
        cb.assignAny(workingArray2, Code.newArray(newEnd)(array.ti))
      })

      cb.assign(i, 0)
      cb.whileLoop(i < newEnd, {
        cb += arrayRef(workingArray1).update(i, array(i))
        cb += arrayRef(workingArray2).update(i, array(i))
        cb.assign(i, i + 1)
      })

      // elements are sorted in workingArray2 after calling splitMergeMB
      cb.invokeVoid(splitMergeMB, sortMB.getCodeParam[Region](1), const(0), newEnd, workingArray1, workingArray2)

      cb.assign(i, 0)
      cb.whileLoop(i < newEnd, {
        cb += array.update(i, arrayRef(workingArray2)(i))
        cb.assign(i, i + 1)
      })

    }
    cb.invokeVoid(sortMB, region)


  }

  def toRegion(cb: EmitCodeBuilder, t: Type): SIndexableCode = {
    t match {
      case pca: TArray =>
        val len = cb.newLocal[Int]("arraysorter_to_region_len", array.size)
        // fixme element requiredness should be set here
        val arrayType = PCanonicalArray(array.elt.loadedSType.canonicalPType().setRequired(this.prunedMissing || array.eltRequired))

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
    this.prunedMissing = true

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

    val distinctMB = cb.emb.genEmitMethod("distinctFromSorted", FastIndexedSeq[ParamType](classInfo[Region]), UnitInfo)
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
