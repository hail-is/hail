package is.hail.expr.ir.ndarrays

import is.hail.annotations.Region
import is.hail.expr.ir._
import is.hail.types.physical.PCanonicalNDArray
import is.hail.types.physical.stypes.interfaces.{SNDArray, SNDArrayCode}
import is.hail.types.physical.stypes.{SCode, SType}
import is.hail.utils._
import is.hail.asm4s._


object EmitNDArray {

  def apply(
    emitter: Emit[_],
    ndIR: IR,
    cb: EmitCodeBuilder,
    region: Value[Region],
    env: Emit.E,
    container: Option[AggContainer]
  ): IEmitCode = {

    def deforest(x: IR): IEmitCodeGen[NDArrayProducer] = {

      def emitI(ir: IR, cb: EmitCodeBuilder, region: Value[Region] = region, env: Emit.E = env, container: Option[AggContainer] = container): IEmitCode = {
        emitter.emitI(ir, cb, region, env, container, None)
      }

      x match {
        case NDArrayMap(child, elemName, body) => {
          deforest(child).map(cb) { childProducer =>
            val elemRef = cb.emb.newPresentEmitField("ndarray_map_element_name", childProducer.elementType)
            val bodyEnv = env.bind(elemName, elemRef)
            val bodyEC = EmitCode.fromI(cb.emb)(cb => emitI(body, cb, env = bodyEnv))

            new NDArrayProducer {
              override def elementType: SType = bodyEC.st

              override val shape: IndexedSeq[Value[Long]] = childProducer.shape
              override val initAll: EmitCodeBuilder => Unit = childProducer.initAll
              override val initAxis: IndexedSeq[EmitCodeBuilder => Unit] = childProducer.initAxis
              override val stepAxis: IndexedSeq[(EmitCodeBuilder, Value[Long]) => Unit] = childProducer.stepAxis

              override def loadElementAtCurrentAddr(cb: EmitCodeBuilder): SCode = {
                cb.assign(elemRef, childProducer.loadElementAtCurrentAddr(cb).toPCode(cb, region))
                bodyEC.toI(cb).get(cb, "NDArray map body cannot be missing")
              }
            }
          }
        }
        case NDArrayReindex(child, indexExpr) =>
          deforest(child).map(cb) { childProducer =>

            new NDArrayProducer {
              override def elementType: SType = childProducer.elementType

              override val shape: IndexedSeq[Value[Long]] = indexExpr.map { childIndex =>
                if (childIndex < childProducer.nDims)
                  childProducer.shape(childIndex)
                else
                  const(1L)
              }
              override val initAll: EmitCodeBuilder => Unit = childProducer.initAll
              override val initAxis: IndexedSeq[EmitCodeBuilder => Unit] = indexExpr.map(childProducer.initAxis.apply)
              override val stepAxis: IndexedSeq[(EmitCodeBuilder, Value[Long]) => Unit] = indexExpr.map(childProducer.stepAxis.apply)

              override def loadElementAtCurrentAddr(cb: EmitCodeBuilder): SCode = childProducer.loadElementAtCurrentAddr(cb)
            }
          }
        case _ => {
          val ndI = emitI(x, cb)
          ndI.map(cb) { ndPCode =>
            val ndPv = ndPCode.asNDArray.memoize(cb, "deforestNDArray_fall_through_ndarray")
            val ndPvShape = ndPv.shapes(cb)
            val strides = ndPv.strides(cb)

            new NDArrayProducer {
              override def elementType: SType = ndPv.st.elementType

              val curOffset = cb.newLocal[Long]("ndarray_producer_fall_through_curr_addr")

              override val shape: IndexedSeq[Value[Long]] = ndPvShape
              override val initAll: EmitCodeBuilder => Unit = cb => cb.assign(curOffset, 0L)
              override val initAxis: IndexedSeq[EmitCodeBuilder => Unit] = {
                shape.indices.map(i => (cb: EmitCodeBuilder) => cb.assign(curOffset, curOffset % strides(i)))
              }
              override val stepAxis: IndexedSeq[(EmitCodeBuilder, Value[Long]) => Unit] = {
                shape.indices.map{ i =>
                  (cb: EmitCodeBuilder, step: Value[Long]) => {
                    cb.assign(curOffset, curOffset + strides(i) * step)
                  }
                }
              }

              override def loadElementAtCurrentAddr(cb: EmitCodeBuilder): SCode = {
                // TODO: Safe to canonicalPType here?
                elementType.loadFrom(cb, region, ndPv.st.elementType.canonicalPType(), ndPv.firstDataAddress(cb) + curOffset)
              }
            }
          }
        }
      }
    }

    deforest(ndIR).map(cb)(ndap => ndap.toSCode(cb, PCanonicalNDArray(ndap.elementType.canonicalPType().setRequired(true), ndap.nDims), region).toPCode(cb, region))
  }
}

abstract class NDArrayProducer {

  def elementType: SType
  val shape: IndexedSeq[Value[Long]]
  def nDims = shape.size

  val initAll: EmitCodeBuilder => Unit
  val initAxis: IndexedSeq[(EmitCodeBuilder) => Unit]
  val stepAxis: IndexedSeq[(EmitCodeBuilder, Value[Long]) => Unit]
  def loadElementAtCurrentAddr(cb: EmitCodeBuilder): SCode

  def toSCode(cb: EmitCodeBuilder, targetType: PCanonicalNDArray, region: Value[Region]): SNDArrayCode =  {
    val (firstElementAddress, finish) = targetType.constructDataFunction(
      shape,
      targetType.makeColumnMajorStrides(shape, region, cb),
      cb,
      region)

    val indices = Array.tabulate(shape.length) { dimIdx => cb.newLocal[Long](s"ndarray_producer_to_scode_foreach_dim_$dimIdx", 0L) }

    def recurLoopBuilder(dimIdx: Int, innerLambda: () => Unit): Unit = {
      if (dimIdx == shape.length) {
        innerLambda()
      }
      else {
        val dimVar = indices(dimIdx)

        recurLoopBuilder(dimIdx + 1,
          () => {
            cb.forLoop({
              initAxis(dimIdx)(cb)
            }, dimVar < shape(dimIdx), {
              stepAxis(dimIdx)(cb, 1L)
              cb.assign(dimVar, dimVar + 1L)
            },
              innerLambda()
            )
          }
        )
      }
    }

    val currentWriteAddr = cb.newLocal[Long]("ndarray_producer_to_scode_cur_write_addr")
    cb.assign(currentWriteAddr, firstElementAddress)
    initAll(cb)
    def body(): Unit = {
      targetType.elementType.storeAtAddress(cb, currentWriteAddr, region, loadElementAtCurrentAddr(cb), true)
      cb.assign(currentWriteAddr, currentWriteAddr + targetType.elementType.byteSize)
    }

    recurLoopBuilder(0, body)

    finish(cb)
  }
}