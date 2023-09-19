package is.hail.expr.ir.streams

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.agg.StagedArrayBuilder
import is.hail.expr.ir.{EmitClassBuilder, EmitCodeBuilder, EmitMethodBuilder, EmitModuleBuilder, SCodeParamType}
import is.hail.types.physical.PCanonicalArray
import is.hail.types.physical.stypes.interfaces.SIndexableValue
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.utils.FastSeq

import scala.language.implicitConversions

sealed trait CGenPriorityQueueClass {
  type ElemType = SValue
  type ArrayType = SIndexableValue
  type This = EmitPriorityQueue.RuntimeClass

  def initialize(cb: EmitCodeBuilder, _this: Settable[This]): Unit

  def realloc(cb: EmitCodeBuilder, _this: Value[This]): Unit

  def close(cb: EmitCodeBuilder, _this: Value[This]): Unit


  def nonEmpty(cb: EmitCodeBuilder, _this: Value[This]): Value[Boolean]

  def peek(cb: EmitCodeBuilder, _this: Value[This]): ElemType

  def poll(cb: EmitCodeBuilder, _this: Value[This]): Unit

  def add(cb: EmitCodeBuilder, _this: Value[This], a: ElemType): Unit

  def toArray(cb: EmitCodeBuilder, _this: Value[This], region: Value[Region]): ArrayType
}

trait StagedFunction2[A, B, C] {
  def initialize(cb: EmitCodeBuilder): Unit
  def apply(cb: EmitCodeBuilder, a: A, b: B): C
  def close(cb: EmitCodeBuilder): Unit =
    ()
}


object EmitPriorityQueue {
  sealed trait RuntimeClass

  def apply(modb: EmitModuleBuilder, elemType: SType)
           (mkComparator: EmitClassBuilder[RuntimeClass]
             => StagedFunction2[SValue, SValue, Value[Int]])
  : CGenPriorityQueueClass = {

    val classBuilder: EmitClassBuilder[RuntimeClass] =
      modb.getOrEmitClass[RuntimeClass](s"CGenPriorityQueue${elemType.asIdent}") { ecb =>
        val region: ThisFieldRef[Region] =
          ecb.genFieldThisRef[Region]("region")

        val garbage: ThisFieldRef[Long] =
          ecb.genFieldThisRef[Long]("n_garbage_points")

        val heap = new StagedArrayBuilder(elemType.storageType(), ecb, region)
        val comparator = mkComparator(ecb)

        ecb.emitInitI { cb =>
          cb.assign(region, cb.emb.ecb.pool().get.invoke[Region]("getRegion"))
          cb.assign(garbage, cb.memoize(0L))
          comparator.initialize(cb)
          heap.initialize(cb)
        }

        val load: EmitMethodBuilder[_] =
          ecb.defineEmitMethod("load", FastSeq(IntInfo), SCodeParamType(elemType)) { mb =>
            mb.emitSCode { cb =>
              heap.loadElement(cb, mb.getCodeParam[Int](0)).toI(cb).get(cb)
            }
          }

        val compareAtIndex: EmitMethodBuilder[_] =
          ecb.defineEmitMethod("compareAtIndex", FastSeq(IntInfo, IntInfo), IntInfo) { mb =>
            mb.emitWithBuilder[Int] { cb =>
              val l = cb.invokeSCode(load, cb._this, mb.getCodeParam[Int](0))
              val r = cb.invokeSCode(load, cb._this, mb.getCodeParam[Int](1))
              comparator.apply(cb, l, r)
            }
          }

        ecb.defineEmitMethod("realloc", FastSeq(), UnitInfo) { mb =>
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

        ecb.defineEmitMethod("close", FastSeq(), UnitInfo) { mb =>
          mb.voidWithBuilder { cb =>
            comparator.close(cb)
            cb += region.invoke[Unit]("invalidate")
          }
        }

        val nonEmpty: EmitMethodBuilder[_] =
          ecb.defineEmitMethod("nonEmpty", FastSeq(), BooleanInfo) { mb =>
            mb.emitWithBuilder { cb =>
              cb.memoize(heap.size > 0)
            }
          }

        ecb.defineEmitMethod("peek", FastSeq(), SCodeParamType(elemType)) { mb =>
          mb.emitSCode { cb =>
            cb += Code._assert(cb.invokeCode[Boolean](nonEmpty, cb._this), s"${ecb.className}: peek empty")
            cb.invokeSCode(load, cb._this, cb.memoize(0))
          }
        }

        val heapify: EmitMethodBuilder[_] =
          ecb.defineEmitMethod("heapify", FastSeq(), UnitInfo) { mb =>
            mb.voidWithBuilder { cb =>
              val index = cb.newLocal[Int]("index", 0)

              val Ldone = CodeLabel()
              val Lentry = CodeLabel()

              cb.define(Lentry)
              cb.ifx(heap.size <= 1, cb.goto(Ldone))

              val child = cb.newLocal[Int]("child")

              cb.assign(child, index * 2 + 1) // left child
              val smallest = cb.newLocal[Int]("smallest", index)
              cb.ifx(child < heap.size && cb.invokeCode[Int](compareAtIndex, cb._this, child, index) < 0, {
                cb.assign(smallest, child)
              })

              cb.assign(child, index * 2 + 2) // right child
              cb.ifx(child < heap.size && cb.invokeCode[Int](compareAtIndex, cb._this, child, smallest) < 0, {
                cb.assign(smallest, child)
              })

              cb.ifx(smallest == index, cb.goto(Ldone))
              heap.swap(cb, index, smallest)
              cb.assign(index, smallest)
              cb.goto(Lentry)

              cb.define(Ldone)
            }
          }

        ecb.defineEmitMethod("poll", FastSeq(), UnitInfo) { mb =>
          mb.voidWithBuilder { cb =>
            cb += Code._assert(cb.invokeCode[Boolean](nonEmpty, cb._this), s"${ecb.className}: poll empty")

            val newSize = cb.memoize(heap.size - 1)
            heap.swap(cb, 0, newSize)
            cb.assign(heap.size, newSize)
            cb.assign(garbage, garbage + 1L)

            cb.invokeVoid(heapify, cb._this)
          }
        }

        ecb.defineEmitMethod("add", FastSeq(SCodeParamType(elemType)), UnitInfo) { mb =>
          mb.voidWithBuilder { cb =>
            val elem = mb.getSCodeParam(1)
            heap.append(cb, elem)

            val Lentry = CodeLabel()
            val Ldone = CodeLabel()

            val current = cb.newLocal[Int]("current", heap.size - 1)

            cb.define(Lentry)
            cb.ifx(current <= 0, cb.goto(Ldone))

            val parent = cb.newLocal[Int]("parent", (current - 1) / 2)
            val cmp = cb.invokeCode[Int](compareAtIndex, cb._this, parent, current)
            cb.ifx(cmp >= 0, cb.goto(Ldone))

            heap.swap(cb, parent, current)
            cb.assign(current, parent)
            cb.goto(Lentry)

            cb.define(Ldone)
          }
        }

        ecb.defineEmitMethod("toArray", FastSeq(SCodeParamType(elemType)), UnitInfo) { mb =>
          mb.voidWithBuilder { cb =>
            PCanonicalArray(elemType.storageType(), required = true)
              .constructFromElements(cb, region, heap.size, true) { case (cb, idx) =>
                heap.loadElement(cb, idx).toI(cb)
              }
          }
        }
      }

    new CGenPriorityQueueClass {
      override def initialize(cb: EmitCodeBuilder, _this: Settable[This]): Unit =
        cb.assign(_this, Code.newInstance(classBuilder.cb, classBuilder.ctor.mb, FastSeq()))

      override def realloc(cb: EmitCodeBuilder, _this: Value[This]): Unit =
        cb.invokeVoid(
          classBuilder.getEmitMethod("realloc", FastSeq(), UnitInfo),
          _this
        )

      override def close(cb: EmitCodeBuilder, _this: Value[This]): Unit =
        cb.invokeVoid(
          classBuilder.getEmitMethod("close", FastSeq(), UnitInfo),
          _this
        )

      override def nonEmpty(cb: EmitCodeBuilder, _this: Value[This]): Value[Boolean] =
        cb.invokeCode(
          classBuilder.getEmitMethod("nonEmpty", FastSeq(), BooleanInfo),
          _this
        )

      override def peek(cb: EmitCodeBuilder, _this: Value[This]): ElemType =
        cb.invokeSCode(
          classBuilder.getEmitMethod("realloc", FastSeq(), elemType.paramType),
          _this
        )

      override def poll(cb: EmitCodeBuilder, _this: Value[This]): Unit =
        cb.invokeVoid(
          classBuilder.getEmitMethod("poll", FastSeq(), UnitInfo),
          _this
        )

      override def add(cb: EmitCodeBuilder, _this: Value[This], a: ElemType): Unit =
        cb.invokeVoid(
          classBuilder.getEmitMethod("add", FastSeq(elemType.paramType), UnitInfo),
          _this,
          a
        )

      override def toArray(cb: EmitCodeBuilder, _this: Value[This], region: Value[Region]): ArrayType =
        cb.invokeSCode(
          classBuilder.getEmitMethod("add", FastSeq(typeInfo[Region]), PCanonicalArray(elemType.storageType()).sType.paramType),
          _this,
          region
        ).asIndexable
    }
  }

}


