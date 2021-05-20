package is.hail.expr.ir.ndarrays

import is.hail.annotations.Region
import is.hail.expr.ir._
import is.hail.types.physical.{PCanonicalArray, PCanonicalNDArray, PInt64}
import is.hail.types.physical.stypes.interfaces.{SNDArray, SNDArrayCode}
import is.hail.types.physical.stypes.{SCode, SType}
import is.hail.utils._
import is.hail.asm4s._
import is.hail.types.physical.stypes.interfaces._


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
                cb.assign(elemRef, childProducer.loadElementAtCurrentAddr(cb))
                bodyEC.toI(cb).get(cb, "NDArray map body cannot be missing")
              }
            }
          }
        }
        case NDArrayMap2(lChild, rChild, lName, rName, body) => {
          deforest(lChild).flatMap(cb) { leftProducer =>
            deforest(rChild).map(cb) { rightProducer =>
              val leftShapeValues = leftProducer.shape
              val rightShapeValues = rightProducer.shape

              val (newSetupShape, shapeArray) = NDArrayEmitter.unifyShapes2(cb.emb, leftShapeValues, rightShapeValues)
              cb.append(newSetupShape)


              val lElemRef = cb.emb.newPresentEmitField(lName, leftProducer.elementType)
              val rElemRef = cb.emb.newPresentEmitField(rName, rightProducer.elementType)
              val bodyEnv = env.bind(lName, lElemRef)
                .bind(rName, rElemRef)
              val bodyEC = EmitCode.fromI(cb.emb)(cb => emitI(body, cb, env = bodyEnv))

              val leftBroadcasted = broadcast(cb, leftProducer, "left")
              val rightBroadcasted = broadcast(cb, rightProducer, "right")

              new NDArrayProducer {
                override def elementType: SType = bodyEC.st

                override val shape: IndexedSeq[Value[Long]] = shapeArray
                override val initAll: EmitCodeBuilder => Unit = {
                  cb => {
                    leftBroadcasted.initAll(cb)
                    rightBroadcasted.initAll(cb)
                  }
                }
                override val initAxis: IndexedSeq[EmitCodeBuilder => Unit] = shape.indices.map { idx =>  { cb: EmitCodeBuilder  =>
                  leftBroadcasted.initAxis(idx)(cb)
                  rightBroadcasted.initAxis(idx)(cb)
                }}
                override val stepAxis: IndexedSeq[(EmitCodeBuilder, Value[Long]) => Unit] = shape.indices.map { idx => { (cb: EmitCodeBuilder, axis: Value[Long]) =>
                  leftBroadcasted.stepAxis(idx)(cb, axis)
                  rightBroadcasted.stepAxis(idx)(cb, axis)
                }}

                override def loadElementAtCurrentAddr(cb: EmitCodeBuilder): SCode = {
                  //cb.println("Load left element")
                  cb.assign(lElemRef, leftBroadcasted.loadElementAtCurrentAddr(cb))
                  //cb.println("Load right element")
                  cb.assign(rElemRef, rightBroadcasted.loadElementAtCurrentAddr(cb))

                  bodyEC.toI(cb).get(cb, "NDArrayMap2 body cannot be missing")
                }
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
              override val initAxis: IndexedSeq[EmitCodeBuilder => Unit] = {
                indexExpr.map { childIndex =>
                  (cb: EmitCodeBuilder) =>
                    if (childIndex < childProducer.nDims) {
                      childProducer.initAxis(childIndex)(cb)
                    }
                }
              }
              override val stepAxis: IndexedSeq[(EmitCodeBuilder, Value[Long]) => Unit] = {
                indexExpr.map { childIndex =>
                  (cb: EmitCodeBuilder, step: Value[Long]) =>
                    if (childIndex < childProducer.nDims) {
                      childProducer.stepAxis(childIndex)(cb, step)
                    }
                }
              }

              override def loadElementAtCurrentAddr(cb: EmitCodeBuilder): SCode = childProducer.loadElementAtCurrentAddr(cb)
            }
          }
        case x@NDArrayReshape(childND,  shape) =>
          deforest(childND).flatMap(cb) { childProducer =>

            val childShapeValues = childProducer.shape
            val outputNDims = x.typ.nDims

            val requestedShapeValues = Array.tabulate(outputNDims)(i => cb.newLocal[Long](s"ndarray_reindex_request_shape_$i")).toIndexedSeq

            emitI(shape, cb, env = env).map(cb) { sc: SCode =>
              val tupleCode = sc.asBaseStruct
              val tupleValue = tupleCode.memoize(cb, "ndarray_reshape_requested")

              val hasNegativeOne = cb.newLocal[Boolean]("ndarray_reshape_has_neg_one")
              val runningProduct = cb.newLocal[Long]("ndarray_reshape_running_product")
              val replacesNegativeOne = cb.newLocal[Long]("ndarray_reshape_replaces_neg_one")
              val tempShapeElement = cb.newLocal[Long]("ndarray_reshape_temp_shape_element")

              cb.assign(hasNegativeOne, false)
              cb.assign(runningProduct, 1L)

              (0 until outputNDims).foreach { i =>
                cb.assign(tempShapeElement, tupleValue.loadField(cb, i).get(cb, "Can't reshape if elements of reshape tuple are missing.").asLong.longCode(cb))
                cb.ifx(tempShapeElement < 0L,
                  {
                    cb.ifx(tempShapeElement ceq -1L,
                      {
                        cb.ifx(hasNegativeOne, {
                          cb._fatal("Can't infer shape, more than one -1")
                        }, {
                          cb.assign(hasNegativeOne, true)
                        })
                      },
                      {
                        cb._fatal("Can't reshape, new shape must contain only nonnegative numbers or -1")
                      }
                    )
                  },
                  {
                    cb.assign(runningProduct, runningProduct * tempShapeElement)
                  }
                )
              }

              val numElements = cb.newLocal[Long]("ndarray_reshape_child_num_elements")
              cb.assign(numElements, SNDArray.numElements(childShapeValues))

              cb.ifx(hasNegativeOne.mux(
                (runningProduct ceq 0L) || (numElements % runningProduct) > 0L,
                numElements cne runningProduct
              ), {
                cb._fatal("Can't reshape since requested shape is incompatible with number of elements")
              })
              cb.assign(replacesNegativeOne, (runningProduct ceq 0L).mux(0L, numElements / runningProduct))

              (0 until outputNDims).foreach { i =>
                cb.assign(tempShapeElement, tupleValue.loadField(cb, i).get(cb, "Can't reshape if elements of reshape tuple are missing.").asLong.longCode(cb))
                cb.assign(requestedShapeValues(i), (tempShapeElement ceq -1L).mux(replacesNegativeOne, tempShapeElement))
              }

              new NDArrayProducer {
                override def elementType: SType = childProducer.elementType

                override val shape: IndexedSeq[Value[Long]] = requestedShapeValues

                override val initAll: EmitCodeBuilder => Unit = childProducer.initAll
                override val initAxis: IndexedSeq[EmitCodeBuilder => Unit] = ???
                override val stepAxis: IndexedSeq[(EmitCodeBuilder, Value[Long]) => Unit] = ???

                override def loadElementAtCurrentAddr(cb: EmitCodeBuilder): SCode = ???
              }
            }
          }

        case x@NDArrayConcat(nds, axis) =>
          emitI(nds, cb).flatMap(cb) { ndsPCode =>
            val ndsArraySValue = ndsPCode.asIndexable.memoize(cb, "ndarray_concat_array_of_nds")
            val arrLength = ndsArraySValue.loadLength()
            cb.ifx(arrLength ceq 0, {
              cb._fatal("need at least one ndarray to concatenate")
            })

            val missing: Code[Boolean] = {
              if (ndsArraySValue.st.elementEmitType.required)
                const(false)
              else {
                val missing = cb.newLocal[Boolean]("ndarray_concat_result_missing")
                cb.assign(missing, false)
                // Need to check if the any of the ndarrays are missing.
                val missingCheckLoopIdx = cb.newLocal[Int]("ndarray_concat_missing_check_idx")
                cb.forLoop(cb.assign(missingCheckLoopIdx, 0), missingCheckLoopIdx < arrLength, cb.assign(missingCheckLoopIdx, missingCheckLoopIdx + 1),
                  cb.assign(missing, missing | ndsArraySValue.isElementMissing(missingCheckLoopIdx))
                )
                missing
              }
            }

            IEmitCode(cb, missing,  {
              val loopIdx = cb.newLocal[Int]("ndarray_concat_shape_check_idx")
              val firstND = ndsArraySValue.loadElement(cb, 0).map(cb) { sCode => sCode.asNDArray }.get(cb).memoize(cb, "ndarray_concat_input_0")

              val stagedArrayOfSizesPType = PCanonicalArray(PInt64(), true)
              val (pushElement, finish) = stagedArrayOfSizesPType.constructFromFunctions(cb, region, arrLength, false)

              val newShape = (0 until x.typ.nDims).map { dimIdx =>
                val localDim = cb.newLocal[Long](s"ndarray_concat_output_shape_element_${ dimIdx }")
                val ndShape = firstND.shapes(cb)
                cb.assign(localDim, ndShape(dimIdx))
                if (dimIdx == axis) {
                  pushElement(cb, EmitCode(Code._empty, false, primitive(localDim)).toI(cb))
                }

                cb.forLoop(cb.assign(loopIdx, 1), loopIdx < arrLength, cb.assign(loopIdx, loopIdx + 1), {
                  val shapeOfNDAtIdx = ndsArraySValue.loadElement(cb, loopIdx).map(cb) { sCode => sCode.asNDArray }.get(cb).shape(cb).memoize(cb, "ndarray_concat_input_shape")
                  val dimLength = cb.newLocal[Long]("dimLength", shapeOfNDAtIdx.loadField(cb, dimIdx).get(cb).asInt64.longCode(cb))

                  if (dimIdx == axis) {
                    pushElement(cb, EmitCode(Code._empty, false, primitive(dimLength)).toI(cb))
                    cb.assign(localDim, localDim + dimLength)
                  }
                  else {
                    cb.ifx(dimLength.cne(localDim),
                      cb._fatal(const(s"NDArrayConcat: mismatched dimensions of input NDArrays along axis ").concat(loopIdx.toS).concat(": expected ")
                        .concat(localDim.toS).concat(", got ")
                        .concat(dimLength.toS))
                    )
                  }
                })
                localDim
              }

              val stagedArrayOfSizes = finish(cb).memoize(cb, "ndarray_concat_staged_array_of_sizes")

              new NDArrayProducer {
                override def elementType: SType = firstND.st.elementType
                override val shape: IndexedSeq[Value[Long]] = newShape

                val idxVars = shape.indices.map(i => cb.newLocal[Long](s"ndarray_produceer_fall_through_idx_${i}"))
                // Need to keep track of the current ndarray being read from.
                val currentNDArrayIdx = cb.newLocal[Int]("ndarray_concat_current_active_ndarray_idx")

                override val initAll: EmitCodeBuilder => Unit = { cb =>
                  idxVars.foreach(idxVar => cb.assign(idxVar, 0L))
                  cb.assign(currentNDArrayIdx, 0)
                }
                override val initAxis: IndexedSeq[EmitCodeBuilder => Unit] =
                  shape.indices.map(i => (cb: EmitCodeBuilder) => {
                    cb.assign(idxVars(i), 0L)
                    if (i == axis) {
                      cb.assign(currentNDArrayIdx, 0)
                    }
                  })
                override val stepAxis: IndexedSeq[(EmitCodeBuilder, Value[Long]) => Unit] = {
                  // For all boring axes, just add to corrsponding indexVar. For the single interesting axis,
                  // also consider updating the currently tracked ndarray.
                  shape.indices.map( idx => (cb: EmitCodeBuilder, step: Value[Long]) => {
                    // Start by updating the idxVar by the step
                    val curIdxVar = idxVars(idx)
                    cb.assign(curIdxVar, curIdxVar + step)
                    if (idx == axis) {
                      // If its too big, panic.
                      val sizeOfCurrentND = cb.newLocal[Long]("ndarray_concat_current_nd_size", stagedArrayOfSizes.loadElement(cb, currentNDArrayIdx).get(cb).asInt64.longCode(cb))
                      cb.ifx(curIdxVar >= sizeOfCurrentND,
                        {
                          cb.assign(curIdxVar, curIdxVar % sizeOfCurrentND)
                          cb.assign(currentNDArrayIdx, currentNDArrayIdx + 1)
                        }
                      )
                    }
                  })
                }

                override def loadElementAtCurrentAddr(cb: EmitCodeBuilder): SCode = {
                  val currentNDArray = ndsArraySValue.loadElement(cb, currentNDArrayIdx).get(cb).asNDArray.memoize(cb, "ndarray_concat_current_active_ndarray")
                  currentNDArray.loadElement(idxVars, cb)
                }
              }
            })
          }
        case _ => {
          val ndI = emitI(x, cb)
          ndI.map(cb) { ndPCode =>
            val ndPv = ndPCode.asNDArray.memoize(cb, "deforestNDArray_fall_through_ndarray")
            val ndPvShape = ndPv.shapes(cb)
            val strides = ndPv.strides(cb)
            val counters = ndPvShape.indices.map(i => cb.newLocal[Long](s"ndarray_produceer_fall_through_idx_${i}"))

            new NDArrayProducer {
              override def elementType: SType = ndPv.st.elementType
              override val shape: IndexedSeq[Value[Long]] = ndPvShape

              override val initAll: EmitCodeBuilder => Unit = cb => {
                counters.foreach(ctr => cb.assign(ctr, 0L))
              }
              override val initAxis: IndexedSeq[EmitCodeBuilder => Unit] = {
                shape.indices.map(i => (cb: EmitCodeBuilder) => {
                  cb.assign(counters(i), 0L)
                })
              }
              override val stepAxis: IndexedSeq[(EmitCodeBuilder, Value[Long]) => Unit] = {
                shape.indices.map{ i =>
                  (cb: EmitCodeBuilder, step: Value[Long]) => {
                    cb.assign(counters(i), counters(i) + step * strides(i))
                  }
                }
              }

              override def loadElementAtCurrentAddr(cb: EmitCodeBuilder): SCode = {
                val offset = counters.foldLeft[Code[Long]](const(0L)){ (a, b) => a + b}
                //cb.println("Fall through Counter values")
                //cb.println(counters.map(_.toS.concat(" ")):_*)
                // TODO: Safe to canonicalPType here?
                val loaded = elementType.loadFrom(cb, region, ndPv.st.elementType.canonicalPType(), ndPv.firstDataAddress(cb) + offset)
                val memoLoaded = loaded.memoize(cb, "temp_memo")
                //cb.println("Looked up ", cb.strValue(memoLoaded))
                memoLoaded.get
              }
            }
          }
        }
      }
    }

    deforest(ndIR).map(cb)(ndap => ndap.toSCode(cb, PCanonicalNDArray(ndap.elementType.canonicalPType().setRequired(true), ndap.nDims), region))
  }

  def createBroadcastMask(cb: EmitCodeBuilder, shape: IndexedSeq[Value[Long]]): IndexedSeq[Value[Long]] = {
    val ffff = 0xFFFFFFFFFFFFFFFFL
    shape.indices.map { idx =>
      cb.newLocal[Long](s"ndarray_producer_broadcast_mask_${idx}", (shape(idx) ceq 1L).mux(0L, ffff))
    }
  }

  def broadcast(cb: EmitCodeBuilder, prod: NDArrayProducer,ctx: String): NDArrayProducer = {
    val broadcastMask = createBroadcastMask(cb, prod.shape)
    //cb.println("Broadcast mask: ", broadcastMask.map(_.toS.concat(" ")).reduce(_ concat _))
    val newSteps = prod.stepAxis.indices.map { idx =>
      (cb: EmitCodeBuilder, step: Value[Long]) => {
        val maskedStep = cb.newLocal[Long]("ndarray_producer_masked_step", step & broadcastMask(idx))
        //cb.println(s"${ctx}: Stepping ", maskedStep.toS, " units on index ", idx.toString)
        prod.stepAxis(idx)(cb, maskedStep)
      }
    }
    prod.copy(astepAxis = newSteps)
  }
}

abstract class NDArrayProducer {
  outer =>

  def elementType: SType
  val shape: IndexedSeq[Value[Long]]
  def nDims = shape.size

  val initAll: EmitCodeBuilder => Unit
  val initAxis: IndexedSeq[(EmitCodeBuilder) => Unit]
  val stepAxis: IndexedSeq[(EmitCodeBuilder, Value[Long]) => Unit]
  def loadElementAtCurrentAddr(cb: EmitCodeBuilder): SCode

  def copy(
    aElementType: SType = elementType,
    aShape: IndexedSeq[Value[Long]] = shape,
    ainitAll: EmitCodeBuilder => Unit = initAll,
    ainitAxis: IndexedSeq[(EmitCodeBuilder) => Unit] = initAxis,
    astepAxis: IndexedSeq[(EmitCodeBuilder, Value[Long]) => Unit] = stepAxis
  ): NDArrayProducer = {
    new NDArrayProducer() {
      override def elementType: SType = aElementType

      override val shape: IndexedSeq[Value[Long]] = aShape
      override val initAll: EmitCodeBuilder => Unit = ainitAll
      override val initAxis: IndexedSeq[EmitCodeBuilder => Unit] = ainitAxis
      override val stepAxis: IndexedSeq[(EmitCodeBuilder, Value[Long]) => Unit] = astepAxis

      override def loadElementAtCurrentAddr(cb: EmitCodeBuilder): SCode = outer.loadElementAtCurrentAddr(cb)
    }
  }

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
              cb.assign(dimVar, 0L)
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
    //   cb.println(const("firstElementAddr = ").concat(firstElementAddress.toS))
    //   cb.println(const("shape =  ").concat(shape.map(_.toS.concat(" ")).reduce(_.concat(_))))
    initAll(cb)
    def body(): Unit = {
      //cb.println(currentWriteAddr.toS)
      targetType.elementType.storeAtAddress(cb, currentWriteAddr, region, loadElementAtCurrentAddr(cb), true)
      //cb.println("Stored successfully")
      cb.assign(currentWriteAddr, currentWriteAddr + targetType.elementType.byteSize)
    }

    recurLoopBuilder(0, body)

    finish(cb)
  }
}