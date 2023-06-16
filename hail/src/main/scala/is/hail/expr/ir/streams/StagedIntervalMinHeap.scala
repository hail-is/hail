package is.hail.expr.ir.streams

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.agg.StagedArrayBuilder
import is.hail.expr.ir.functions.IntervalFunctions
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, EmitMethodBuilder, ParamType}
import is.hail.types.physical.stypes.concrete.SStackStruct
import is.hail.types.physical.{PBaseStruct, PCanonicalArray, PCanonicalTuple, PInterval}
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.physical.stypes.interfaces.{SBaseStruct, SBaseStructValue, SContainer, SIndexableValue, SInterval, SIntervalValue}
import is.hail.types.virtual.TTuple
import is.hail.utils.FastIndexedSeq

// min heap on interval right endpoints
class StagedIntervalMinHeap(mb: EmitMethodBuilder[_],
  inputTupleType: SBaseStruct) {

  val tupleType = inputTupleType.copiedType.storageType().asInstanceOf[PBaseStruct]


  val region = mb.genFieldThisRef[Region]("region")
  val tmpTuple = mb.genFieldThisRef[Long]("tmp_tuple")
  val mostRecentlyAddedIntervalData = mb.genFieldThisRef[Long]("tmp_tuple")

  val garbage = mb.genFieldThisRef[Long]("n_garbage_points")
  val maxSizeSoFar = mb.genFieldThisRef[Long]("max_size")

  val resultArrayType: PCanonicalArray = PCanonicalArray(tupleType, true)
  val resultArraySType: SContainer = resultArrayType.sType


  val ab = new StagedArrayBuilder(tupleType, mb.ecb, region)

  def nodeAtIdx(cb: EmitCodeBuilder, idx: Value[Int]): SValue = ab.loadElement(cb, cb.memoize(idx - 1)).toI(cb).get(cb)

  def intervalAtIndex(cb: EmitCodeBuilder, idx: Value[Int]): SIntervalValue = nodeAtIdx(cb, idx).asBaseStruct.loadField(cb, 0).get(cb).asInterval

  def initialize(cb: EmitCodeBuilder): Unit = {
    ab.initialize(cb)
    cb.assign(region, cb.emb.ecb.pool().get.invoke[Region]("getRegion"))
    cb.assign(tmpTuple, region.invoke[Long, Long, Long]("allocate", tupleType.byteSize, 8L))
    cb.assign(mostRecentlyAddedIntervalData, region.invoke[Long, Long, Long]("allocate", tupleType.types(0).byteSize, 8L))
  }

  def swap(cb: EmitCodeBuilder, _l: Value[Int], _r: Value[Int]): Unit = {
    val l = cb.memoize(_l - 1)
    val r = cb.memoize(_r - 1)
    cb += Region.copyFrom(ab.elementOffset(cb, l), tmpTuple, tupleType.byteSize)
    cb += Region.copyFrom(ab.elementOffset(cb, r), ab.elementOffset(cb, l), tupleType.byteSize)
    cb += Region.copyFrom(tmpTuple, ab.elementOffset(cb, r), tupleType.byteSize)
  }

  def loadIntervalFromIdx(cb: EmitCodeBuilder, idx: Code[Int]): SIntervalValue = {
    ab.loadElement(cb, cb.memoize(idx + 1))
      .toI(cb)
      .get(cb)
      .asBaseStruct
      .loadField(cb, 0)
      .get(cb)
      .asInterval
  }

  val compareAtIndex:  (EmitCodeBuilder, Value[Int], Value[Int]) => Code[Int] = {

    val method = mb.ecb.newEmitMethod("lt_at_idx", FastIndexedSeq[ParamType](IntInfo, IntInfo), UnitInfo)

    mb.emitWithBuilder { cb =>
      val l = loadIntervalFromIdx(cb, method.getCodeParam[Int](0))
      val r = loadIntervalFromIdx(cb, method.getCodeParam[Int](1))
      IntervalFunctions.intervalEndpointCompare(cb, l.loadEnd(cb).get(cb), l.includesEnd, r.loadEnd(cb).get(cb), r.includesEnd)
    }

    { (cb: EmitCodeBuilder, l: Value[Int], r: Value[Int]) => method.invokeCode[Int](cb, l, r) }
  }

  val bubbleDown: (EmitCodeBuilder, Value[Int]) => Unit = {

    val method = mb.ecb.newEmitMethod("bubble_down", FastIndexedSeq[ParamType](IntInfo), UnitInfo)

    method.voidWithBuilder { cb =>
      val idx = method.getCodeParam[Int](0)

      val test1 = cb.memoize(idx * 2)
      cb.ifx(test1 <= ab.size, {
        val test2 = cb.memoize(idx * 2 + 1)
        val idxToCompare = cb.newLocal[Int]("idx_to_compare", test1)
        cb.ifx(test2 <= ab.size && compareAtIndex(cb, test1, test2) < 0,
          cb.assign(idxToCompare, test2))

        cb.ifx(compareAtIndex(cb, idx, idxToCompare) > 0, {
          swap(cb, idx, cb.memoize(idxToCompare))
          cb.invokeVoid(method, idxToCompare)
        })
      })
    }

    { (cb: EmitCodeBuilder, idx: Value[Int]) => cb.invokeVoid(method, idx) }
  }

  val bubbleUp: (EmitCodeBuilder, Value[Int]) => Unit = {

    val method = mb.ecb.newEmitMethod("bubble_up", FastIndexedSeq[ParamType](IntInfo), UnitInfo)

    method.voidWithBuilder { cb =>
      val idx = cb.memoize(method.getCodeParam[Int](0))
      cb.ifx(idx > 1, {
        val test = cb.memoize(idx / 2)
        cb.ifx(compareAtIndex(cb, idx, test) < 0, {
          swap(cb, idx, test)
          cb.invokeVoid(method, test)
        })
      })
    }

    { (cb: EmitCodeBuilder, idx: Value[Int]) => cb.invokeVoid(method, idx) }
  }

  // drops contained intervals whose right endpoints are not contained by `point`
  def dropLessThan(cb: EmitCodeBuilder, point: SValue): Unit = {
    cb.whileLoop( ab.size > 0 && {
      val firstInterval = loadIntervalFromIdx(cb, 1)
      IntervalFunctions.pointGTIntervalEndpoint(cb, point, firstInterval.loadEnd(cb).get(cb), firstInterval.includesEnd)
    }, {
      swap(cb, 1, ab.size)
      cb.assign(ab.size, ab.size - 1)
      bubbleDown(cb, 1)
      cb.assign(garbage, garbage + 1)
    })

    // needs to do GC

  }

  // adds an interval and data tuple to the priority queue
  def addInterval(cb: EmitCodeBuilder, tuple: SBaseStructValue): Unit = {
    ab.append(cb, tuple)
    // copy memory for most recent interval since the memory in the builder is mutable and we can't save a pointer
    cb += Region.copyFrom(tupleType.loadField(ab.elementOffset(cb, cb.memoize(ab.size - 1)), 0), mostRecentlyAddedIntervalData, tupleType.types(0).byteSize)

    bubbleUp(cb, ab.size)


  }

  def getAllContainedIntervalsAsArray(cb: EmitCodeBuilder, resultRegion: Value[Region]): SIndexableValue = {
    resultArrayType.constructFromElements(cb, resultRegion, ab.size, true) { case (cb, idx) =>
      ab.loadElement(cb, idx).toI(cb)
    }
  }

  def size: Value[Int] = ab.size

  // used
  def mostRecentlyAddedInterval(cb: EmitCodeBuilder): SIntervalValue = tupleType.types(0).loadCheapSCode(cb, mostRecentlyAddedIntervalData).asInterval

  def close(cb: EmitCodeBuilder): Unit = {
    cb += region.invoke[Unit]("invalidate")
  }
}
