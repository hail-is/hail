package is.hail.expr.ir.streams

import is.hail.annotations.{Region, RegionPool}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitCodeBuilder, EmitMethodBuilder, EmitModuleBuilder}
import is.hail.expr.ir.agg.StagedArrayBuilder
import is.hail.types.physical.PCanonicalArray
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.physical.stypes.interfaces.SIndexableValue
import is.hail.utils.FastSeq

sealed trait StagedMinHeap {
  def arraySType: SType

  def init(cb: EmitCodeBuilder, pool: Value[RegionPool]): Unit
  def realloc(cb: EmitCodeBuilder): Unit
  def close(cb: EmitCodeBuilder): Unit

  def push(cb: EmitCodeBuilder, a: SValue): Unit
  def peek(cb: EmitCodeBuilder): SValue
  def pop(cb: EmitCodeBuilder): Unit
  def nonEmpty(cb: EmitCodeBuilder): Value[Boolean]

  def toArray(cb: EmitCodeBuilder, region: Value[Region]): SIndexableValue
}

object StagedMinHeap {
  def apply(
    modb: EmitModuleBuilder,
    elemSType: SType,
  )(
    comparator: (EmitCodeBuilder, SValue, SValue) => Value[Int]
  ): EmitClassBuilder[_] => StagedMinHeap = {

    val elemPType = elemSType.storageType().setRequired(true)
    val elemParamType = elemPType.sType.paramType

    val classBuilder: EmitClassBuilder[Unit] =
      modb.genEmitClass[Unit](s"MinHeap${elemPType.asIdent}")

    val pool: ThisFieldRef[RegionPool] =
      classBuilder.genFieldThisRef[RegionPool]("pool")

    val region: ThisFieldRef[Region] =
      classBuilder.genFieldThisRef[Region]("region")

    val garbage: ThisFieldRef[Long] =
      classBuilder.genFieldThisRef[Long]("n_garbage_points")

    val heap = new StagedArrayBuilder(elemPType, classBuilder, region)
    val ctor: EmitMethodBuilder[Unit] =
      classBuilder.defineEmitMethod("<init>", FastSeq(typeInfo[RegionPool]), UnitInfo) { mb =>
        val poolRef = mb.getCodeParam[RegionPool](1)

        mb.voidWithBuilder { cb =>
          cb += classBuilder.cb.super_.invoke(coerce[Object](cb.this_), Array())
          cb.assign(pool, poolRef)
          cb.assign(region, Region.stagedCreate(Region.REGULAR, poolRef))
          cb.assign(garbage, 0L)
          heap.initialize(cb)
        }
      }

    val load: EmitMethodBuilder[_] =
      classBuilder.defineEmitMethod("load", FastSeq(IntInfo), elemParamType) { mb =>
        mb.emitSCode { cb =>
          val idx = mb.getCodeParam[Int](1)
          heap.loadElement(cb, idx).toI(cb).getOrAssert(cb, debugMsg = idx.toS)
        }
      }

    val compareAtIndex: EmitMethodBuilder[_] =
      classBuilder.defineEmitMethod("compareAtIndex", FastSeq(IntInfo, IntInfo), IntInfo) { mb =>
        mb.emitWithBuilder[Int] { cb =>
          val l = cb.invokeSCode(load, cb.this_, mb.getCodeParam[Int](1))
          val r = cb.invokeSCode(load, cb.this_, mb.getCodeParam[Int](2))
          comparator(cb, l, r)
        }
      }

    val realloc_ : EmitMethodBuilder[_] =
      classBuilder.defineEmitMethod("realloc", FastSeq(), UnitInfo) { mb =>
        mb.voidWithBuilder { cb =>
          cb.if_(
            garbage > heap.size.toL * 2L + 1024L, {
              val oldRegion = cb.memoize(region, "tmp")
              cb.assign(region, Region.stagedCreate(Region.REGULAR, pool))
              heap.reallocateData(cb)
              cb.assign(garbage, 0L)
              cb += oldRegion.invoke[Unit]("invalidate")
            },
          )
        }
      }

    val close_ : EmitMethodBuilder[_] =
      classBuilder.defineEmitMethod("close", FastSeq(), UnitInfo) { mb =>
        mb.emit {
          region.invoke[Unit]("invalidate")
        }
      }

    def thisNonEmpty: Code[Boolean] =
      heap.size > 0

    val peek_ : EmitMethodBuilder[_] =
      classBuilder.defineEmitMethod("peek", FastSeq(), elemParamType) { mb =>
        mb.emitSCode { cb =>
          cb._assert(thisNonEmpty, s"${classBuilder.className}: peek empty")
          cb.invokeSCode(load, cb.this_, cb.memoize(0))
        }
      }

    val swap: EmitMethodBuilder[_] =
      classBuilder.defineEmitMethod("swap", FastSeq(IntInfo, IntInfo), UnitInfo) { mb =>
        mb.voidWithBuilder { cb =>
          val x = mb.getCodeParam[Int](1)
          val y = mb.getCodeParam[Int](2)
          heap.swap(cb, x, y)
        }
      }

    val heapify: EmitMethodBuilder[_] =
      classBuilder.defineEmitMethod("heapify", FastSeq(), UnitInfo) { mb =>
        mb.voidWithBuilder { cb =>
          val Ldone = CodeLabel()
          cb.if_(heap.size <= 1, cb.goto(Ldone))

          val index = cb.newLocal[Int]("index", 0)
          val smallest = cb.newLocal[Int]("smallest", index)

          val child = cb.newLocal[Int]("child")
          cb.loop { Lrecur =>
            // left child
            cb.assign(child, index * 2 + 1)
            cb.if_(
              child < heap.size,
              cb.if_(
                cb.invokeCode[Int](compareAtIndex, cb.this_, child, index) < 0,
                cb.assign(smallest, child),
              ),
            )

            // right child
            cb.assign(child, index * 2 + 2)
            cb.if_(
              child < heap.size,
              cb.if_(
                cb.invokeCode[Int](compareAtIndex, cb.this_, child, smallest) < 0,
                cb.assign(smallest, child),
              ),
            )

            cb.if_(smallest ceq index, cb.goto(Ldone))

            cb.invokeVoid(swap, cb.this_, index, smallest)
            cb.assign(index, smallest)
            cb.goto(Lrecur)
          }

          cb.define(Ldone)
        }
      }

    val pop_ : EmitMethodBuilder[_] =
      classBuilder.defineEmitMethod("pop", FastSeq(), UnitInfo) { mb =>
        mb.voidWithBuilder { cb =>
          cb._assert(thisNonEmpty, s"${classBuilder.className}: poll empty")

          cb.assign(garbage, garbage + 1L)
          val newSize = cb.memoize(heap.size - 1)
          cb.if_(
            newSize ceq 0,
            cb.assign(heap.size, newSize), {
              cb.invokeVoid(swap, cb.this_, const(0), newSize)
              cb.assign(heap.size, newSize)
              cb.invokeVoid(heapify, cb.this_)
            },
          )
        }
      }

    val append: EmitMethodBuilder[_] =
      classBuilder.defineEmitMethod("append", FastSeq(elemParamType), UnitInfo) { mb =>
        mb.voidWithBuilder(cb => heap.append(cb, mb.getSCodeParam(1)))
      }

    val push_ : EmitMethodBuilder[_] =
      classBuilder.defineEmitMethod("push", FastSeq(elemParamType), UnitInfo) { mb =>
        mb.voidWithBuilder { cb =>
          cb.invokeVoid(append, cb.this_, mb.getSCodeParam(1))
          val Ldone = CodeLabel()
          val current = cb.newLocal[Int]("index", heap.size - 1)
          val parent = cb.newLocal[Int]("parent")

          cb.while_(
            current > 0, {
              cb.assign(parent, (current - 1) / 2)
              val cmp = cb.invokeCode[Int](compareAtIndex, cb.this_, parent, current)
              cb.if_(cmp <= 0, cb.goto(Ldone))

              cb.invokeVoid(swap, cb.this_, parent, current)
              cb.assign(current, parent)
            },
          )

          cb.define(Ldone)
        }
      }

    val arrayPType = PCanonicalArray(elemPType, required = true)

    val toArray_ : EmitMethodBuilder[_] =
      classBuilder.defineEmitMethod(
        "toArray",
        FastSeq(typeInfo[Region]),
        arrayPType.sType.paramType,
      ) { mb =>
        val region = mb.getCodeParam[Region](1)
        mb.emitSCode { cb =>
          arrayPType.constructFromElements(cb, region, heap.size, true) {
            case (cb, idx) => heap.loadElement(cb, idx).toI(cb)
          }
        }
      }

    ecb =>
      new StagedMinHeap {
        private[this] val this_ : ThisFieldRef[_] =
          ecb.genFieldThisRef("minheap")(classBuilder.cb.ti)

        override def arraySType: SType =
          arrayPType.sType

        override def init(cb: EmitCodeBuilder, pool: Value[RegionPool]): Unit =
          cb.assignAny(this_, Code.newInstance(classBuilder.cb, ctor.mb, FastSeq(pool)))

        override def realloc(cb: EmitCodeBuilder): Unit =
          cb.invokeVoid(realloc_, this_)

        override def close(cb: EmitCodeBuilder): Unit =
          cb.invokeVoid(close_, this_)

        override def push(cb: EmitCodeBuilder, a: SValue): Unit =
          cb.invokeVoid(push_, this_, a)

        override def peek(cb: EmitCodeBuilder): SValue =
          cb.invokeSCode(peek_, this_)

        override def pop(cb: EmitCodeBuilder): Unit =
          cb.invokeVoid(pop_, this_)

        override def nonEmpty(cb: EmitCodeBuilder): Value[Boolean] =
          cb.memoize(classBuilder.getField[Int](heap.size.name).get(this_) > 0)

        override def toArray(cb: EmitCodeBuilder, region: Value[Region]): SIndexableValue =
          cb.invokeSCode(toArray_, this_, region).asIndexable
      }
  }
}
