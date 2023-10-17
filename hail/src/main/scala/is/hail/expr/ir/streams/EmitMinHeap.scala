package is.hail.expr.ir.streams

import is.hail.annotations.{Region, RegionPool}
import is.hail.asm4s._
import is.hail.expr.ir.agg.StagedArrayBuilder
import is.hail.expr.ir.{EmitClassBuilder, EmitCodeBuilder, EmitMethodBuilder, EmitModuleBuilder, SCodeParamType}
import is.hail.types.physical.PCanonicalArray
import is.hail.types.physical.stypes.interfaces.SIndexableValue
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.utils.FastSeq

sealed trait StagedMinHeap {
  
  def init(cb: EmitCodeBuilder, pool: Value[RegionPool]): Unit

  def realloc(cb: EmitCodeBuilder): Unit

  def close(cb: EmitCodeBuilder): Unit

  def push(cb: EmitCodeBuilder, a: SValue): Unit
  def peek(cb: EmitCodeBuilder): SValue
  def pop(cb: EmitCodeBuilder): Unit
  def nonEmpty(cb: EmitCodeBuilder): Value[Boolean]

  def toArray(cb: EmitCodeBuilder, region: Value[Region]): SIndexableValue
}

object EmitMinHeap {

  private sealed trait MinHeap

  def apply(modb: EmitModuleBuilder, elemType: SType)
           (mkComparator: EmitClassBuilder[_] => StagedComparator)
  : EmitClassBuilder[_] => StagedMinHeap = {

    val classBuilder: EmitClassBuilder[MinHeap] =
      modb.genEmitClass[MinHeap](s"MinHeap${elemType.asIdent}")

    val pool: ThisFieldRef[RegionPool] =
      classBuilder.genFieldThisRef[RegionPool]("pool")

    val region: ThisFieldRef[Region] =
      classBuilder.genFieldThisRef[Region]("region")

    val garbage: ThisFieldRef[Long] =
      classBuilder.genFieldThisRef[Long]("n_garbage_points")

    val heap = new StagedArrayBuilder(elemType.storageType(), classBuilder, region)
    val comparator = mkComparator(classBuilder)

    val init_ : EmitMethodBuilder[MinHeap] =
      classBuilder.defineEmitMethod("<init>", FastSeq(typeInfo[AnyRef], typeInfo[RegionPool]), UnitInfo) { mb =>
        val super_ = Invokeable(classOf[Object], classOf[Object].getConstructor())
        val outerRef = mb.getCodeParam[AnyRef](1)
        val poolRef = mb.getCodeParam[RegionPool](2)

        mb.voidWithBuilder { cb =>
          cb += super_.invoke(coerce[Object](cb._this), Array())
          cb.assign(pool, poolRef)
          cb.assign(region, poolRef.invoke[Region]("getRegion"))
          cb.assign(garbage, cb.memoize(0L))
          comparator.init(cb, outerRef)
          heap.initialize(cb)
        }
      }

    val load: EmitMethodBuilder[_] =
      classBuilder.defineEmitMethod("load", FastSeq(IntInfo), SCodeParamType(elemType)) { mb =>
        mb.emitSCode { cb =>
          val idx = mb.getCodeParam[Int](1)
          heap.loadElement(cb, idx).toI(cb).get(cb, errorMsg = idx.toS)
        }
      }

    val compareAtIndex: EmitMethodBuilder[_] =
      classBuilder.defineEmitMethod("compareAtIndex", FastSeq(IntInfo, IntInfo), IntInfo) { mb =>
        mb.emitWithBuilder[Int] { cb =>
          val l = cb.invokeSCode(load, cb._this, mb.getCodeParam[Int](1))
          val r = cb.invokeSCode(load, cb._this, mb.getCodeParam[Int](2))
          comparator.apply(cb, l, r)
        }
      }

    val realloc_ : EmitMethodBuilder[_] =
      classBuilder.defineEmitMethod("realloc", FastSeq(), UnitInfo) { mb =>
        mb.voidWithBuilder { cb =>
          cb.ifx(garbage > heap.size.toL * 2L + 1024L, {
            val oldRegion = cb.newLocal[Region]("tmp", region)
            cb.assign(region, pool.invoke[Region]("getRegion"))
            heap.reallocateData(cb)
            cb.assign(garbage, 0L)
            cb += oldRegion.invoke[Unit]("invalidate")
          })
        }
      }

    val close_ : EmitMethodBuilder[_] =
      classBuilder.defineEmitMethod("close", FastSeq(), UnitInfo) { mb =>
        mb.voidWithBuilder { cb =>
          comparator.close(cb)
          cb += region.invoke[Unit]("invalidate")
        }
      }

    val nonEmpty_ : EmitMethodBuilder[_] =
      classBuilder.defineEmitMethod("nonEmpty", FastSeq(), BooleanInfo) { mb =>
        mb.emitWithBuilder { cb =>
          cb.memoize(heap.size > 0)
        }
      }

    val peek_ : EmitMethodBuilder[_] =
      classBuilder.defineEmitMethod("peek", FastSeq(), SCodeParamType(elemType)) { mb =>
        mb.emitSCode { cb =>
          cb += Code._assert(cb.invokeCode[Boolean](nonEmpty_, cb._this), s"${classBuilder.className}: peek empty")
          cb.invokeSCode(load, cb._this, cb.memoize(0))
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
          cb.ifx(heap.size <= 1, cb.goto(Ldone))

          val index = cb.newLocal[Int]("index", 0)
          val smallest = cb.newLocal[Int]("smallest", index)

          val child = cb.newLocal[Int]("child")
          cb.loop { Lrecur =>
            // left child
            cb.assign(child, index * 2 + 1)
            cb.ifx(child < heap.size,
              cb.ifx(
                cb.invokeCode[Int](compareAtIndex, cb._this, child, index) < 0,
                cb.assign(smallest, child)
              )
            )

            // right child
            cb.assign(child, index * 2 + 2)
            cb.ifx(child < heap.size,
              cb.ifx(
                cb.invokeCode[Int](compareAtIndex, cb._this, child, smallest) < 0,
                cb.assign(smallest, child)
              )
            )

            cb.ifx(smallest ceq index, cb.goto(Ldone))

            cb.invokeVoid(swap, cb._this, index, smallest)
            cb.assign(index, smallest)
            cb.goto(Lrecur)
          }

          cb.define(Ldone)
        }
      }

    val pop_ : EmitMethodBuilder[_] =
      classBuilder.defineEmitMethod("pop", FastSeq(), UnitInfo) { mb =>
        mb.voidWithBuilder { cb =>
          cb += Code._assert(cb.invokeCode[Boolean](nonEmpty_, cb._this), s"${classBuilder.className}: poll empty")

          val newSize = cb.memoize(heap.size - 1)
          cb.invokeVoid(swap, cb._this, const(0), newSize)
          cb.assign(heap.size, newSize)
          cb.assign(garbage, garbage + 1L)
          cb.invokeVoid(heapify, cb._this)
        }
      }

    val append: EmitMethodBuilder[_] =
      classBuilder.defineEmitMethod("append", FastSeq(SCodeParamType(elemType)), UnitInfo) { mb =>
        mb.voidWithBuilder { cb =>
          heap.append(cb, mb.getSCodeParam(1))
        }
      }

    val push_ : EmitMethodBuilder[_] =
      classBuilder.defineEmitMethod("push", FastSeq(SCodeParamType(elemType)), UnitInfo) { mb =>
        mb.voidWithBuilder { cb =>
          cb.invokeVoid(append, cb._this, mb.getSCodeParam(1))

          val Ldone = CodeLabel()
          val current = cb.newLocal[Int]("index", heap.size - 1)
          val parent = cb.newLocal[Int]("parent")

          cb.whileLoop(current > 0, {
            cb.assign(parent, (current - 1) / 2)
            val cmp = cb.invokeCode[Int](compareAtIndex, cb._this, parent, current)
            cb.ifx(cmp <= 0, cb.goto(Ldone))

            cb.invokeVoid(swap, cb._this, parent, current)
            cb.assign(current, parent)
          })

          cb.define(Ldone)
        }
      }

    val arrayTy = PCanonicalArray(elemType.storageType(), required = true)

    val toArray_ : EmitMethodBuilder[_] =
      classBuilder.defineEmitMethod("toArray", FastSeq(typeInfo[Region]), arrayTy.sType.paramType) { mb =>
        val region = mb.getCodeParam[Region](1)
        mb.emitSCode { cb =>
          arrayTy.constructFromElements(cb, region, heap.size, true) {
            case (cb, idx) => heap.loadElement(cb, idx).toI(cb)
          }
        }
      }

    ecb => new StagedMinHeap {
      private[this] val _this: ThisFieldRef[_] =
        ecb.genFieldThisRef("minheap")(classBuilder.cb.ti)

      override def init(cb: EmitCodeBuilder, pool: Value[RegionPool]): Unit =
        cb.assignAny(_this,
          Code.newInstance(classBuilder.cb, init_.mb, FastSeq(
            cb.memoize(Code.checkcast[AnyRef](cb._this)), // `this` of the parent class
            pool
          ))
        )

      override def realloc(cb: EmitCodeBuilder): Unit =
        cb.invokeVoid(realloc_, _this)

      override def close(cb: EmitCodeBuilder): Unit =
        cb.invokeVoid(close_, _this)

      override def push(cb: EmitCodeBuilder, a: SValue): Unit =
        cb.invokeVoid(push_, _this, a)

      override def peek(cb: EmitCodeBuilder): SValue =
        cb.invokeSCode(peek_, _this)

      override def pop(cb: EmitCodeBuilder): Unit =
        cb.invokeVoid(pop_, _this)

      override def nonEmpty(cb: EmitCodeBuilder): Value[Boolean] =
        cb.invokeCode(nonEmpty_, _this)

      override def toArray(cb: EmitCodeBuilder, region: Value[Region]): SIndexableValue =
        cb.invokeSCode(toArray_, _this, region).asIndexable
    }
  }

  trait StagedComparator {
    def init(cb: EmitCodeBuilder, enclosingRef: Value[AnyRef]): Unit = ()

    def apply(cb: EmitCodeBuilder, a: SValue, b: SValue): Value[Int]

    def close(cb: EmitCodeBuilder): Unit = ()
  }

}


