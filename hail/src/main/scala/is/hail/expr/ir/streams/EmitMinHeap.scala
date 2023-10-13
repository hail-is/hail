package is.hail.expr.ir.streams

import is.hail.annotations.{Region, RegionPool}
import is.hail.asm4s._
import is.hail.expr.ir.agg.StagedArrayBuilder
import is.hail.expr.ir.{EmitClassBuilder, EmitCodeBuilder, EmitMethodBuilder, EmitModuleBuilder, SCodeParamType}
import is.hail.types.physical.PCanonicalArray
import is.hail.types.physical.stypes.interfaces.SIndexableValue
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.utils.FastSeq

import scala.language.implicitConversions

sealed trait StagedMinHeap {
  
  def initialize(cb: EmitCodeBuilder, pool: Value[RegionPool]): Unit

  def realloc(cb: EmitCodeBuilder): Unit

  def close(cb: EmitCodeBuilder): Unit


  def nonEmpty(cb: EmitCodeBuilder): Value[Boolean]

  def peek(cb: EmitCodeBuilder): SValue

  def poll(cb: EmitCodeBuilder): Unit

  def add(cb: EmitCodeBuilder, a: SValue): Unit

  def toArray(cb: EmitCodeBuilder, region: Value[Region]): SIndexableValue
}

object EmitMinHeap {

  def apply(modb: EmitModuleBuilder, elemType: SType)
           (mkComparator: EmitClassBuilder[_] => StagedComparator)
  : EmitClassBuilder[_] => StagedMinHeap = {
    sealed trait MinHeap

    val classBuilder: EmitClassBuilder[MinHeap] =
      modb.getOrEmitClass[MinHeap](s"MinHeap_${elemType.asIdent}") { kb =>
        val pool: ThisFieldRef[RegionPool] =
          kb.genFieldThisRef[RegionPool]("pool")

        val region: ThisFieldRef[Region] =
          kb.genFieldThisRef[Region]("region")

        val garbage: ThisFieldRef[Long] =
          kb.genFieldThisRef[Long]("n_garbage_points")

        val heap = new StagedArrayBuilder(elemType.storageType(), kb, region)
        val comparator = mkComparator(kb)

        // `RegionPool`s are set added after construction therefore need a separate init method.
        // See `EmitClassBuilder.resultWithIndex`.
        kb.defineEmitMethod("init", FastSeq(typeInfo[AnyRef], typeInfo[RegionPool]), UnitInfo) { mb =>
          val outerRef = mb.getCodeParam[AnyRef](1)
          val poolRef = mb.getCodeParam[RegionPool](2)

          mb.voidWithBuilder { cb =>
            cb.assign(pool, poolRef)
            cb.assign(region, poolRef.invoke[Region]("getRegion"))
            cb.assign(garbage, cb.memoize(0L))
            comparator.initialize(cb, outerRef)
            heap.initialize(cb)
          }
        }

        val load: EmitMethodBuilder[_] =
          kb.defineEmitMethod("load", FastSeq(IntInfo), SCodeParamType(elemType)) { mb =>
            mb.emitSCode { cb =>
              val idx = mb.getCodeParam[Int](1)
              heap.loadElement(cb, idx).toI(cb).get(cb)
            }
          }

        val compareAtIndex: EmitMethodBuilder[_] =
          kb.defineEmitMethod("compareAtIndex", FastSeq(IntInfo, IntInfo), IntInfo) { mb =>
            mb.emitWithBuilder[Int] { cb =>
              val l = cb.invokeSCode(load, cb._this, mb.getCodeParam[Int](1))
              val r = cb.invokeSCode(load, cb._this, mb.getCodeParam[Int](2))
              comparator.apply(cb, l, r)
            }
          }

        kb.defineEmitMethod("realloc", FastSeq(), UnitInfo) { mb =>
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

        kb.defineEmitMethod("close", FastSeq(), UnitInfo) { mb =>
          mb.voidWithBuilder { cb =>
            comparator.close(cb)
            cb += region.invoke[Unit]("invalidate")
          }
        }

        val nonEmpty: EmitMethodBuilder[_] =
          kb.defineEmitMethod("nonEmpty", FastSeq(), BooleanInfo) { mb =>
            mb.emitWithBuilder { cb =>
              cb.memoize(heap.size > 0)
            }
          }

        kb.defineEmitMethod("peek", FastSeq(), SCodeParamType(elemType)) { mb =>
          mb.emitSCode { cb =>
            cb += Code._assert(cb.invokeCode[Boolean](nonEmpty, cb._this), s"${kb.className}: peek empty")
            cb.invokeSCode(load, cb._this, cb.memoize(0))
          }
        }

        val heapify: EmitMethodBuilder[_] =
          kb.defineEmitMethod("heapify", FastSeq(), UnitInfo) { mb =>
            mb.voidWithBuilder { cb =>
              val index = cb.newLocal[Int]("index", 0)

              val Ldone = CodeLabel()
              cb.ifx(heap.size <= 1, cb.goto(Ldone))

              cb.loop { Lrecur =>
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

                cb.ifx(smallest ceq index, cb.goto(Ldone))
                heap.swap(cb, index, smallest)
                cb.assign(index, smallest)
                cb.goto(Lrecur)
              }

              cb.define(Ldone)
            }
          }

        kb.defineEmitMethod("poll", FastSeq(), UnitInfo) { mb =>
          mb.voidWithBuilder { cb =>
            cb += Code._assert(cb.invokeCode[Boolean](nonEmpty, cb._this), s"${kb.className}: poll empty")

            val newSize = cb.memoize(heap.size - 1)
            heap.swap(cb, 0, newSize)
            cb.assign(heap.size, newSize)
            cb.assign(garbage, garbage + 1L)

            cb.invokeVoid(heapify, cb._this)
          }
        }

        kb.defineEmitMethod("add", FastSeq(SCodeParamType(elemType)), UnitInfo) { mb =>
          mb.voidWithBuilder { cb =>
            val elem = mb.getSCodeParam(1)
            heap.append(cb, elem)

            val Ldone = CodeLabel()
            val current = cb.newLocal[Int]("current", heap.size - 1)
            cb.ifx(current <= 1, cb.goto(Ldone))

            val parent = cb.newLocal[Int]("parent")
            cb.loop { Lrecur =>
              cb.assign(parent, (current - 1) / 2)
              val cmp = cb.invokeCode[Int](compareAtIndex, cb._this, parent, current)
              cb.ifx(cmp >= 0, cb.goto(Ldone))

              heap.swap(cb, parent, current)
              cb.assign(current, parent)
              cb.goto(Lrecur)
            }

            cb.define(Ldone)
          }
        }


        val arrayTy = PCanonicalArray(elemType.storageType(), required = true)
        kb.defineEmitMethod("toArray", FastSeq(typeInfo[Region]), arrayTy.sType.paramType) { mb =>
          val region = mb.getCodeParam[Region](1)
          mb.emitSCode { cb =>
            arrayTy.constructFromElements(cb, region, heap.size, true) {
              case (cb, idx) => heap.loadElement(cb, idx).toI(cb)
            }
          }
        }
      }

    ecb => new StagedMinHeap {
      private[this] val _this: ThisFieldRef[_] =
        ecb.genFieldThisRef("minheap")(classBuilder.cb.ti)

      override def initialize(cb: EmitCodeBuilder, pool: Value[RegionPool]): Unit = {
        cb.assignAny(_this, Code.newInstance(classBuilder.cb, classBuilder.ctor.mb, FastSeq()))
        cb.invokeVoid(
          classBuilder.getEmitMethod("init", FastSeq(typeInfo[AnyRef], typeInfo[RegionPool]), UnitInfo),
          _this,
          cb.memoize(Code.checkcast[AnyRef](cb._this)), // `this` of the parent class
          pool
        )
      }

      override def realloc(cb: EmitCodeBuilder): Unit =
        cb.invokeVoid(
          classBuilder.getEmitMethod("realloc", FastSeq(), UnitInfo),
          _this
        )

      override def close(cb: EmitCodeBuilder): Unit =
        cb.invokeVoid(
          classBuilder.getEmitMethod("close", FastSeq(), UnitInfo),
          _this
        )

      override def nonEmpty(cb: EmitCodeBuilder): Value[Boolean] =
        cb.invokeCode(
          classBuilder.getEmitMethod("nonEmpty", FastSeq(), BooleanInfo),
          _this
        )

      override def peek(cb: EmitCodeBuilder): SValue =
        cb.invokeSCode(
          classBuilder.getEmitMethod("peek", FastSeq(), elemType.paramType),
          _this
        )

      override def poll(cb: EmitCodeBuilder): Unit =
        cb.invokeVoid(
          classBuilder.getEmitMethod("poll", FastSeq(), UnitInfo),
          _this
        )

      override def add(cb: EmitCodeBuilder, a: SValue): Unit =
        cb.invokeVoid(
          classBuilder.getEmitMethod("add", FastSeq(elemType.paramType), UnitInfo),
          _this,
          a
        )

      override def toArray(cb: EmitCodeBuilder, region: Value[Region]): SIndexableValue =
        cb.invokeSCode(
          classBuilder.getEmitMethod("toArray", FastSeq(typeInfo[Region]), PCanonicalArray(elemType.storageType()).sType.paramType),
          _this,
          region
        ).asIndexable
    }
  }

  trait StagedComparator {
    def initialize(cb: EmitCodeBuilder, enclosingRef: Value[AnyRef]): Unit = ()

    def apply(cb: EmitCodeBuilder, a: SValue, b: SValue): Value[Int]

    def close(cb: EmitCodeBuilder): Unit = ()
  }

}


