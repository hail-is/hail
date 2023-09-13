package is.hail.expr.ir.streams

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.agg.StagedArrayBuilder
import is.hail.expr.ir.functions.IntervalFunctions
import is.hail.expr.ir.{CodeParamType, EmitClassBuilder, EmitCode, EmitCodeBuilder, EmitFunctionBuilder, EmitMethodBuilder, EmitModuleBuilder, Param, ParamType, SCodeParamType}
import is.hail.types.physical.stypes.concrete.{SIntervalPointer, SStackStruct}
import is.hail.types.physical.{PBaseStruct, PCanonicalArray, PCanonicalTuple, PInterval}
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.physical.stypes.interfaces.{SBaseStruct, SBaseStructValue, SContainer, SIndexableValue, SInterval, SIntervalValue}
import is.hail.types.physical.stypes.primitives.SBooleanValue
import is.hail.types.virtual.TTuple
import is.hail.utils.{FastIndexedSeq, FastSeq}


sealed trait StagedPriorityQueue {
  def initialize(cb: EmitCodeBuilder): Unit

  def realloc(cb: EmitCodeBuilder): Unit

  def close(cb: EmitCodeBuilder): Unit


  def nonEmpty(cb: EmitCodeBuilder): Value[Boolean]

  def add(cb: EmitCodeBuilder, v: SValue): Unit

  def peek(cb: EmitCodeBuilder): SValue

  def poll(cb: EmitCodeBuilder): Unit


  def toArray(cb: EmitCodeBuilder, region: Value[Region]): SIndexableValue
}


object StagedPriorityQueue {

  def apply(modb: EmitModuleBuilder, elemType: SType, compare: EmitFunctionBuilder[Unit]): StagedPriorityQueue = {

    val kb = modb.getOrEmitNewClass[Unit]("StagedPriorityQueue") { kb =>

      val region: ThisFieldRef[Region] =
        kb.genFieldThisRef[Region]("region")

      val garbage: ThisFieldRef[Long] =
        kb.genFieldThisRef[Long]("n_garbage_points")

      val heap =
        new StagedArrayBuilder(elemType.storageType(), kb, region)

      val load: EmitMethodBuilder[Unit] =
        kb.getOrGenEmitMethod("load", "load", FastSeq(IntInfo), SCodeParamType(elemType)) { mb =>
          mb.emitSCode { cb =>
            heap.loadElement(cb, mb.getCodeParam[Int](0)).toI(cb).get(cb)
          }
        }

      val compareAtIndex: EmitMethodBuilder[Unit] =
        kb.getOrGenEmitMethod("compareAtIndex", "compareAtIndex", FastSeq(IntInfo, IntInfo), IntInfo) { mb =>
          mb.emitWithBuilder[Int] { cb =>
            val l = cb.invokeSCode(load, mb.getCodeParam[Int](0))
            val r = cb.invokeSCode(load, mb.getCodeParam[Int](1))
            cb.invokeCode[Int](compare.emb, l, r)
          }
        }

      val heapify: EmitMethodBuilder[Unit] =
        kb.getOrGenEmitMethod("heapify", "heapify", Array.empty, UnitInfo) { mb =>
          mb.voidWithBuilder { cb =>
            val index = cb.newLocal[Int]("index", 0)

            val Ldone = CodeLabel()
            val Lentry = CodeLabel()

            cb.define(Lentry)
            cb.ifx(heap.size <= 1, cb.goto(Ldone))

            val left = cb.newLocal[Int]("leftChild", 2 * index + 1)
            val smallest = cb.newLocal[Int]("smallest", index)
            cb.ifx(left < heap.size && cb.invokeCode[Boolean](compareAtIndex, left, index) < 0, {
              cb.assign(smallest, left)
            })

            val right = cb.newLocal[Int]("rightChild", 2 * index + 2)
            cb.ifx(right < heap.size && cb.invokeCode[Boolean](compareAtIndex, right, smallest) < 0, {
              cb.assign(smallest, right)
            })

            cb.ifx(smallest == index, cb.goto(Ldone))
            heap.swap(cb, index, smallest)
            cb.assign(index, smallest)
            cb.goto(Lentry)

            cb.define(Ldone)
          }
        }

      kb.getOrGenEmitMethod("initialize", ("initialize", kb), Array.empty, UnitInfo) { mb =>
        mb.voidWithBuilder { cb =>
          cb.assign(region, cb.emb.ecb.pool().get.invoke[Region]("getRegion"))
          heap.initialize(cb)
        }
      }

      kb.getOrGenEmitMethod("realloc", "realloc", Array.empty, UnitInfo) { mb =>
        mb.voidWithBuilder { cb =>
          cb.ifx(garbage > heap.size.toL * 2L + 1024L, {
            val oldRegion = cb.newLocal[Region]("tmp", region)
            cb.assign(region, cb.emb.ecb.pool().get.invoke[Region]("getRegion"))
            heap.reallocateData(cb)
            cb.assign(garbage, 0L)
            cb += oldRegion.invoke[Unit]("invalidate")
          })
        }
      }

      kb.getOrGenEmitMethod("close", "close", Array.empty, UnitInfo) { mb =>
        mb.voidWithBuilder { cb =>
          cb += region.invoke[Unit]("invalidate")
        }
      }

      val nonEmpty: EmitMethodBuilder[Unit] =
        kb.getOrGenEmitMethod("nonEmpty", "nonEmpty", Array.empty, BooleanInfo) { mb =>
          mb.emitWithBuilder { cb =>
            cb.memoize(heap.size > 0)
          }
        }

      kb.getOrGenEmitMethod("peek", "peek", Array.empty, SCodeParamType(elemType)) { mb =>
        mb.emitSCode { cb =>
          cb += Code._assert(cb.invokeCode[Boolean](nonEmpty), "peek empty StagedPriorityQueue")
          cb.invokeSCode(load, cb.memoize(0))
        }
      }

      kb.getOrGenEmitMethod("poll", "poll", Array.empty, UnitInfo) { mb =>
        mb.voidWithBuilder { cb =>
          cb += Code._assert(cb.invokeCode[Boolean](nonEmpty), "poll empty StagedPriorityQueue")

          val newSize = cb.memoize(heap.size - 1)
          heap.swap(cb, 0, newSize)
          cb.assign(heap.size, newSize)
          cb.assign(garbage, garbage + 1)

          cb.invokeVoid(heapify)
        }
      }

      kb.getOrGenEmitMethod("add", "add", FastSeq(SCodeParamType(elemType)), UnitInfo) { mb =>
        mb.voidWithBuilder { cb =>
          val elem = mb.getSCodeParam(1)
          heap.append(cb, elem)

          val Lentry = CodeLabel()
          val Ldone = CodeLabel()

          val current = cb.newLocal[Int]("current", heap.size - 1)

          cb.define(Lentry)
          cb.ifx(current <= 0, cb.goto(Ldone))

          val parent = cb.newLocal[Int]("parent", (current - 1) / 2)
          val cmp = cb.invokeCode[Int](compareAtIndex, parent, current)
          cb.ifx(cmp >= 0, cb.goto(Ldone))

          heap.swap(cb, parent, current)
          cb.assign(current, parent)
          cb.goto(Lentry)

          cb.define(Ldone)
        }
      }

      kb.getOrGenEmitMethod("toArray", "toArray", FastSeq(SCodeParamType(elemType)), UnitInfo) { mb =>
        mb.voidWithBuilder { cb =>
          PCanonicalArray(elemType.storageType(), required = true)
            .constructFromElements(cb, region, heap.size, true) { case (cb, idx) =>
              heap.loadElement(cb, idx).toI(cb)
            }
        }
      }
    }

    new StagedPriorityQueue {
      override def initialize(cb: EmitCodeBuilder): Unit =
        cb.invokeVoid(kb.getEmitMethod("initialize"))

      override def realloc(cb: EmitCodeBuilder): Unit =
        cb.invokeVoid(kb.getEmitMethod("realloc"))

      override def close(cb: EmitCodeBuilder): Unit =
        cb.invokeVoid(kb.getEmitMethod("close"))

      override def nonEmpty(cb: EmitCodeBuilder): Value[Boolean] =
        cb.invokeCode[Boolean](kb.getEmitMethod("nonEmpty"))

      override def peek(cb: EmitCodeBuilder): SValue =
        cb.invokeSCode(kb.getEmitMethod("peek"))

      override def poll(cb: EmitCodeBuilder): Unit =
        cb.invokeVoid(kb.getEmitMethod("poll", "poll", Array.empty, UnitInfo))

      override def add(cb: EmitCodeBuilder, v: SValue): Unit =
        cb.invokeVoid(kb.getEmitMethod("add"), v)

      override def toArray(cb: EmitCodeBuilder, region: Value[Region]): SIndexableValue =
        cb.invokeSCode(kb.getEmitMethod("toArray"), region).asIndexable
    }
  }
}
