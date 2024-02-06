package is.hail.expr.ir.ndarrays

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir._
import is.hail.types.physical._
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.virtual._
import is.hail.utils._

abstract class NDArrayProducer {
  outer =>

  def elementType: PType
  val shape: IndexedSeq[Value[Long]]
  def nDims = shape.size

  val initAll: EmitCodeBuilder => Unit
  val initAxis: IndexedSeq[(EmitCodeBuilder) => Unit]
  val stepAxis: IndexedSeq[(EmitCodeBuilder, Value[Long]) => Unit]
  def loadElementAtCurrentAddr(cb: EmitCodeBuilder): SValue

  def copy(
    aElementType: PType = elementType,
    aShape: IndexedSeq[Value[Long]] = shape,
    ainitAll: EmitCodeBuilder => Unit = initAll,
    ainitAxis: IndexedSeq[(EmitCodeBuilder) => Unit] = initAxis,
    astepAxis: IndexedSeq[(EmitCodeBuilder, Value[Long]) => Unit] = stepAxis,
  ): NDArrayProducer = {
    new NDArrayProducer() {
      override def elementType: PType = aElementType

      override val shape: IndexedSeq[Value[Long]] = aShape
      override val initAll: EmitCodeBuilder => Unit = ainitAll
      override val initAxis: IndexedSeq[EmitCodeBuilder => Unit] = ainitAxis
      override val stepAxis: IndexedSeq[(EmitCodeBuilder, Value[Long]) => Unit] = astepAxis

      override def loadElementAtCurrentAddr(cb: EmitCodeBuilder): SValue =
        outer.loadElementAtCurrentAddr(cb)
    }
  }

  def toSCode(
    cb: EmitCodeBuilder,
    targetType: PCanonicalNDArray,
    region: Value[Region],
    rowMajor: Boolean = false,
  ): SNDArrayValue = {
    val (firstElementAddress, finish) = targetType.constructDataFunction(
      shape,
      targetType.makeColumnMajorStrides(shape, cb),
      cb,
      region,
    )

    val currentWriteAddr = cb.newLocal[Long]("ndarray_producer_to_scode_cur_write_addr")
    cb.assign(currentWriteAddr, firstElementAddress)

    initAll(cb)
    val idxGenerator = if (rowMajor) SNDArray.forEachIndexWithInitAndIncRowMajor _
    else SNDArray.forEachIndexWithInitAndIncColMajor _
    idxGenerator(
      cb,
      shape,
      initAxis,
      stepAxis.map(stepper => (cb: EmitCodeBuilder) => stepper(cb, 1L)),
      "ndarray_producer_toSCode",
    ) { (cb, indices) =>
      targetType.elementType.storeAtAddress(
        cb,
        currentWriteAddr,
        region,
        loadElementAtCurrentAddr(cb),
        true,
      )
      cb.assign(currentWriteAddr, currentWriteAddr + targetType.elementType.byteSize)
    }

    finish(cb)
  }
}

object EmitNDArray {

  def apply(
    emitter: Emit[_],
    ndIR: IR,
    cb: EmitCodeBuilder,
    region: Value[Region],
    env: EmitEnv,
    container: Option[AggContainer],
    loopEnv: Option[Env[LoopRef]],
  ): IEmitCode = {

    def emitNDInSeparateMethod(
      context: String,
      cb: EmitCodeBuilder,
      ir: IR,
      region: Value[Region],
      env: EmitEnv,
      container: Option[AggContainer],
      loopEnv: Option[Env[LoopRef]],
    ): IEmitCode = {

      assert(!emitter.ctx.inLoopCriticalPath.contains(ir))
      val mb = cb.emb.genEmitMethod(context, FastSeq[ParamType](), UnitInfo)
      val r = cb.newField[Region]("emitInSeparate_region", region)

      var ev: EmitSettable = null
      mb.voidWithBuilder { cb =>
        emitter.ctx.tryingToSplit.update(ir, ())
        val result: IEmitCode = deforest(ir, cb, r, env, container, loopEnv).map(cb)(ndap =>
          ndap.toSCode(cb, PCanonicalNDArray(ndap.elementType.setRequired(true), ndap.nDims), r)
        )

        ev = cb.emb.ecb.newEmitField(s"${context}_result", result.emitType)
        cb.assign(ev, result)
      }
      cb.invokeVoid(mb, cb.this_)
      ev.toI(cb)
    }

    def deforest(
      x: IR,
      cb: EmitCodeBuilder,
      region: Value[Region],
      env: EmitEnv,
      container: Option[AggContainer],
      loopEnv: Option[Env[LoopRef]],
    ): IEmitCodeGen[NDArrayProducer] = {
      def deforestRecur(
        x: IR,
        cb: EmitCodeBuilder = cb,
        region: Value[Region] = region,
        env: EmitEnv = env,
        container: Option[AggContainer] = container,
        loopEnv: Option[Env[LoopRef]] = loopEnv,
      ): IEmitCodeGen[NDArrayProducer] = {

        def emitI(
          ir: IR,
          cb: EmitCodeBuilder,
          region: Value[Region] = region,
          env: EmitEnv = env,
          container: Option[AggContainer] = container,
          loopEnv: Option[Env[LoopRef]] = loopEnv,
        ): IEmitCode =
          emitter.emitI(ir, cb, region, env, container, loopEnv)

        x match {
          case NDArrayMap(child, elemName, body) =>
            deforestRecur(child, cb).map(cb) { childProducer =>
              val elemRef = cb.emb.newEmitField(
                "ndarray_map_element_name",
                childProducer.elementType.sType,
                required = true,
              )
              val bodyEnv = env.bind(elemName, elemRef)
              val bodyEC = EmitCode.fromI(cb.emb)(cb => emitI(body, cb, env = bodyEnv))

              new NDArrayProducer {
                override def elementType: PType = bodyEC.st.storageType()

                override val shape: IndexedSeq[Value[Long]] = childProducer.shape
                override val initAll: EmitCodeBuilder => Unit = childProducer.initAll
                override val initAxis: IndexedSeq[EmitCodeBuilder => Unit] = childProducer.initAxis
                override val stepAxis: IndexedSeq[(EmitCodeBuilder, Value[Long]) => Unit] =
                  childProducer.stepAxis

                override def loadElementAtCurrentAddr(cb: EmitCodeBuilder): SValue = {
                  cb.assign(
                    elemRef,
                    EmitCode.present(cb.emb, childProducer.loadElementAtCurrentAddr(cb)),
                  )
                  bodyEC.toI(cb).get(cb, "NDArray map body cannot be missing")
                }
              }
            }
          case NDArrayMap2(lChild, rChild, lName, rName, body, errorID) =>
            deforestRecur(lChild, cb).flatMap(cb) { leftProducer =>
              deforestRecur(rChild, cb).map(cb) { rightProducer =>
                val leftShapeValues = leftProducer.shape
                val rightShapeValues = rightProducer.shape

                val shapeArray =
                  NDArrayEmitter.unifyShapes2(cb, leftShapeValues, rightShapeValues, errorID)

                val lElemRef =
                  cb.emb.newEmitField(lName, leftProducer.elementType.sType, required = true)
                val rElemRef =
                  cb.emb.newEmitField(rName, rightProducer.elementType.sType, required = true)
                val bodyEnv = env.bind(lName, lElemRef)
                  .bind(rName, rElemRef)
                val bodyEC = EmitCode.fromI(cb.emb)(cb => emitI(body, cb, env = bodyEnv))

                val leftBroadcasted = broadcast(cb, leftProducer, "left")
                val rightBroadcasted = broadcast(cb, rightProducer, "right")

                new NDArrayProducer {
                  override def elementType: PType = bodyEC.st.storageType()

                  override val shape: IndexedSeq[Value[Long]] = shapeArray
                  override val initAll: EmitCodeBuilder => Unit = {
                    cb =>
                      leftBroadcasted.initAll(cb)
                      rightBroadcasted.initAll(cb)
                  }
                  override val initAxis: IndexedSeq[EmitCodeBuilder => Unit] =
                    shape.indices.map { idx => cb: EmitCodeBuilder =>
                      leftBroadcasted.initAxis(idx)(cb)
                      rightBroadcasted.initAxis(idx)(cb)
                    }
                  override val stepAxis: IndexedSeq[(EmitCodeBuilder, Value[Long]) => Unit] =
                    shape.indices.map { idx => (cb: EmitCodeBuilder, axis: Value[Long]) =>
                      leftBroadcasted.stepAxis(idx)(cb, axis)
                      rightBroadcasted.stepAxis(idx)(cb, axis)
                    }

                  override def loadElementAtCurrentAddr(cb: EmitCodeBuilder): SValue = {
                    cb.assign(
                      lElemRef,
                      EmitCode.present(cb.emb, leftBroadcasted.loadElementAtCurrentAddr(cb)),
                    )
                    cb.assign(
                      rElemRef,
                      EmitCode.present(cb.emb, rightBroadcasted.loadElementAtCurrentAddr(cb)),
                    )

                    bodyEC.toI(cb).get(cb, "NDArrayMap2 body cannot be missing", errorID)
                  }
                }
              }
            }
          case NDArrayReindex(child, indexExpr) =>
            deforestRecur(child, cb).map(cb) { childProducer =>
              new NDArrayProducer {
                override def elementType: PType = childProducer.elementType

                override val shape: IndexedSeq[Value[Long]] = indexExpr.map { childIndex =>
                  if (childIndex < childProducer.nDims)
                    childProducer.shape(childIndex)
                  else
                    const(1L)
                }
                override val initAll: EmitCodeBuilder => Unit = childProducer.initAll
                override val initAxis: IndexedSeq[EmitCodeBuilder => Unit] =
                  indexExpr.map { childIndex => (cb: EmitCodeBuilder) =>
                    if (childIndex < childProducer.nDims) {
                      childProducer.initAxis(childIndex)(cb)
                    }
                  }
                override val stepAxis: IndexedSeq[(EmitCodeBuilder, Value[Long]) => Unit] =
                  indexExpr.map { childIndex => (cb: EmitCodeBuilder, step: Value[Long]) =>
                    if (childIndex < childProducer.nDims) {
                      childProducer.stepAxis(childIndex)(cb, step)
                    }
                  }

                override def loadElementAtCurrentAddr(cb: EmitCodeBuilder): SValue =
                  childProducer.loadElementAtCurrentAddr(cb)
              }
            }
          case x @ NDArrayReshape(childND, shape, errorID) =>
            emitI(childND, cb).flatMap(cb) { case childND: SNDArrayValue =>
              /* Plan: Run through the child row major, make an array. Then jump around it as
               * needed. */
              val childShapeValues = childND.shapes
              val outputNDims = x.typ.nDims

              val requestedShapeValues = Array.tabulate(outputNDims)(i =>
                cb.newLocal[Long](s"ndarray_reindex_request_shape_$i")
              ).toIndexedSeq

              emitI(shape, cb, env = env).map(cb) { case tupleValue: SBaseStructValue =>
                val hasNegativeOne = cb.newLocal[Boolean]("ndarray_reshape_has_neg_one")
                val runningProduct = cb.newLocal[Long]("ndarray_reshape_running_product")
                val replacesNegativeOne = cb.newLocal[Long]("ndarray_reshape_replaces_neg_one")
                val tempShapeElement = cb.newLocal[Long]("ndarray_reshape_temp_shape_element")

                cb.assign(hasNegativeOne, false)
                cb.assign(runningProduct, 1L)

                (0 until outputNDims).foreach { i =>
                  cb.assign(
                    tempShapeElement,
                    tupleValue.loadField(cb, i).get(
                      cb,
                      "Can't reshape if elements of reshape tuple are missing.",
                      errorID,
                    ).asLong.value,
                  )
                  cb.if_(
                    tempShapeElement < 0L, {
                      cb.if_(
                        tempShapeElement ceq -1L,
                        cb.if_(
                          hasNegativeOne,
                          cb._fatalWithError(errorID, "Can't infer shape, more than one -1"),
                          cb.assign(hasNegativeOne, true),
                        ),
                        cb._fatalWithError(
                          errorID,
                          "Can't reshape, new shape must contain only nonnegative numbers or -1",
                        ),
                      )
                    },
                    cb.assign(runningProduct, runningProduct * tempShapeElement),
                  )
                }

                val numElements = cb.newLocal[Long]("ndarray_reshape_child_num_elements")
                cb.assign(numElements, SNDArray.numElements(childShapeValues))

                cb.if_(
                  hasNegativeOne.mux(
                    (runningProduct ceq 0L) || (numElements % runningProduct) > 0L,
                    numElements cne runningProduct,
                  ),
                  cb._fatalWithError(
                    errorID,
                    "Can't reshape since requested shape is incompatible with number of elements",
                  ),
                )
                cb.assign(
                  replacesNegativeOne,
                  (runningProduct ceq 0L).mux(0L, numElements / runningProduct),
                )

                (0 until outputNDims).foreach { i =>
                  cb.assign(
                    tempShapeElement,
                    tupleValue.loadField(cb, i).get(
                      cb,
                      "Can't reshape if elements of reshape tuple are missing.",
                      errorID,
                    ).asLong.value,
                  )
                  cb.assign(
                    requestedShapeValues(i),
                    (tempShapeElement ceq -1L).mux(replacesNegativeOne, tempShapeElement),
                  )
                }

                val childPType = childND.st.storageType().asInstanceOf[PCanonicalNDArray]
                val rowMajor = fromSValue(childND, cb).toSCode(cb, childPType, region, true)
                /* The canonical row major thing is now in the order we want. We just need to read
                 * this with the row major striding that */
                // would be generated for something of the new shape.
                val outputPType = PCanonicalNDArray(
                  rowMajor.st.elementPType.setRequired(true),
                  x.typ.nDims,
                  true,
                ) // TODO Should it be required?
                val rowMajorStriding = outputPType.makeRowMajorStrides(requestedShapeValues, cb)
                fromShapeStridesFirstAddress(
                  rowMajor.st.elementPType,
                  requestedShapeValues,
                  rowMajorStriding,
                  rowMajor.firstDataAddress,
                  cb,
                )
              }
            }

          case x @ NDArrayConcat(nds, axis) =>
            emitI(nds, cb).flatMap(cb) { case ndsArraySValue: SIndexableValue =>
              val arrLength = ndsArraySValue.loadLength()
              cb.if_(arrLength ceq 0, cb._fatal("need at least one ndarray to concatenate"))

              IEmitCode(
                cb,
                ndsArraySValue.hasMissingValues(cb), {
                  val firstND = ndsArraySValue.loadElement(cb, 0).get(cb).asNDArray

                  // compute array of sizes along concat axis, and total size of concat axis
                  val arrayLongPType = PCanonicalArray(PInt64(), true)
                  val (pushSize, finishSizes) =
                    arrayLongPType.constructFromFunctions(cb, region, arrLength, false)

                  val concatAxisSize = cb.newLocal[Long](s"ndarray_concat_axis_size", 0L)

                  ndsArraySValue.forEachDefined(cb) { (cb, i, nd) =>
                    val dimLength = nd.asNDArray.shapes(axis)
                    pushSize(cb, EmitCode(Code._empty, false, primitive(dimLength)).toI(cb))
                    cb.assign(concatAxisSize, concatAxisSize + dimLength)
                  }
                  val stagedArrayOfSizes = finishSizes(cb)

                  // compute index of first input which has non-zero concat axis size
                  val firstNonEmpty = cb.newLocal[Int]("ndarray_concat_first_nonempty", 0)
                  cb.while_(
                    stagedArrayOfSizes.loadElement(cb, firstNonEmpty).get(cb).asInt64.value.ceq(0L),
                    cb.assign(firstNonEmpty, firstNonEmpty + 1),
                  )

                  // check that all sizes along other axes are consistent
                  ndsArraySValue.forEachDefined(cb) { case (cb, _, nd: SNDArrayValue) =>
                    val mismatchedDim = cb.newLocal[Int]("ndarray_concat_mismatched_dim", -1)
                    val expected = cb.newLocal[Long]("ndarray_concat_expected_size")
                    val found = cb.newLocal[Long]("ndarray_concat_found_size")
                    for (i <- (0 until firstND.st.nDims).reverse if i != axis)
                      cb.if_(
                        firstND.shapes(i).cne(nd.shapes(i)), {
                          cb.assign(mismatchedDim, i)
                          cb.assign(expected, firstND.shapes(i))
                          cb.assign(found, nd.shapes(i))
                        },
                      )
                    cb.if_(
                      mismatchedDim.cne(-1),
                      cb._fatal(
                        const(s"NDArrayConcat: mismatched dimensions of input NDArrays along axis ")
                          .concat(mismatchedDim.toS)
                          .concat(": expected ").concat(expected.toS)
                          .concat(", got ").concat(found.toS)
                      ),
                    )
                  }

                  // compute shape of result
                  val newShape = (0 until x.typ.nDims).map { i =>
                    if (i == axis) concatAxisSize else firstND.shapes(i)
                  }

                  new NDArrayProducer {
                    override def elementType: PType = firstND.st.elementPType

                    override val shape: IndexedSeq[Value[Long]] = newShape

                    val idxVars = shape.indices.map(i =>
                      cb.newLocal[Long](s"ndarray_produceer_fall_through_idx_$i")
                    )
                    // Need to keep track of the current ndarray being read from.
                    val currentNDArrayIdx =
                      cb.newLocal[Int]("ndarray_concat_current_active_ndarray_idx")

                    override val initAll: EmitCodeBuilder => Unit = { cb =>
                      idxVars.foreach(idxVar => cb.assign(idxVar, 0L))
                      cb.assign(currentNDArrayIdx, firstNonEmpty)
                    }
                    override val initAxis: IndexedSeq[EmitCodeBuilder => Unit] =
                      shape.indices.map(i =>
                        (cb: EmitCodeBuilder) => {
                          cb.assign(idxVars(i), 0L)
                          if (i == axis) {
                            cb.assign(currentNDArrayIdx, firstNonEmpty)
                          }
                        }
                      )
                    override val stepAxis: IndexedSeq[(EmitCodeBuilder, Value[Long]) => Unit] = {
                      /* For all boring axes, just add to corresponding indexVar. For the single
                       * interesting axis, */
                      // also consider updating the currently tracked ndarray.
                      shape.indices.map(idx =>
                        (cb: EmitCodeBuilder, step: Value[Long]) => {
                          // Start by updating the idxVar by the step
                          val curIdxVar = idxVars(idx)
                          cb.assign(curIdxVar, curIdxVar + step)
                          if (idx == axis) {
                            /* If bigger than current ndarray, then we need to subtract out the size
                             * of this ndarray, increment to the next ndarray, and see if we are
                             * happy yet. */
                            val shouldLoop = cb.newLocal[Boolean](
                              "should_loop",
                              curIdxVar >= stagedArrayOfSizes.loadElement(
                                cb,
                                currentNDArrayIdx,
                              ).get(cb).asInt64.value,
                            )
                            cb.while_(
                              shouldLoop, {
                                cb.assign(
                                  curIdxVar,
                                  curIdxVar - stagedArrayOfSizes.loadElement(
                                    cb,
                                    currentNDArrayIdx,
                                  ).get(cb).asInt64.value,
                                )
                                cb.assign(currentNDArrayIdx, currentNDArrayIdx + 1)
                                cb.if_(
                                  currentNDArrayIdx < stagedArrayOfSizes.loadLength(),
                                  cb.assign(
                                    shouldLoop,
                                    curIdxVar >= stagedArrayOfSizes.loadElement(
                                      cb,
                                      currentNDArrayIdx,
                                    ).get(cb).asInt64.value,
                                  ),
                                  cb.assign(shouldLoop, false),
                                )
                              },
                            )
                          }
                        }
                      )
                    }

                    override def loadElementAtCurrentAddr(cb: EmitCodeBuilder): SValue = {
                      val currentNDArray =
                        ndsArraySValue.loadElement(cb, currentNDArrayIdx).get(cb).asNDArray
                      currentNDArray.loadElement(idxVars, cb)
                    }
                  }
                },
              )
            }
          case NDArraySlice(child, slicesIR) =>
            deforestRecur(child, cb).flatMap(cb) { childProducer =>
              emitI(slicesIR, cb).flatMap(cb) { case slicesValue: SBaseStructValue =>
                val (indexingIndices, slicingIndices) =
                  slicesValue.st.fieldTypes.zipWithIndex.partition { case (pFieldType, _) =>
                    pFieldType.isPrimitive
                  } match {
                    case (a, b) => (a.map(_._2), b.map(_._2))
                  }

                IEmitCode.multiFlatMap[Int, SValue, NDArrayProducer](
                  indexingIndices,
                  indexingIndex => slicesValue.loadField(cb, indexingIndex),
                  cb,
                ) { indexingSCodes =>
                  val indexingValues = indexingSCodes.map(sCode =>
                    cb.newLocal("ndarray_slice_indexer", sCode.asInt64.value)
                  )
                  val slicingValueTriplesBuilder =
                    new BoxedArrayBuilder[(Value[Long], Value[Long], Value[Long])]()
                  val outputShape = {
                    IEmitCode.multiFlatMap[Int, SValue, IndexedSeq[Value[Long]]](
                      slicingIndices,
                      valueIdx => slicesValue.loadField(cb, valueIdx),
                      cb,
                    ) { sCodeSlices: IndexedSeq[SValue] =>
                      IEmitCode.multiFlatMap[SValue, Value[Long], IndexedSeq[Value[Long]]](
                        sCodeSlices,
                        { case sValueSlice: SBaseStructValue =>
                          // I know I have a tuple of three elements here, start, stop, step

                          val newDimSizeI = sValueSlice.loadField(cb, 0).flatMap(cb) { startC =>
                            sValueSlice.loadField(cb, 1).flatMap(cb) { stopC =>
                              sValueSlice.loadField(cb, 2).map(cb) { stepC =>
                                val start = startC.asLong.value
                                val stop = stopC.asLong.value
                                val step = stepC.asLong.value

                                slicingValueTriplesBuilder.push((start, stop, step))

                                val newDimSize = cb.newLocal[Long]("new_dim_size")
                                cb.if_(
                                  step >= 0L && start <= stop,
                                  cb.assign(newDimSize, const(1L) + ((stop - start) - 1L) / step),
                                  cb.if_(
                                    step < 0L && start >= stop,
                                    cb.assign(newDimSize, (((stop - start) + 1L) / step) + 1L),
                                    cb.assign(newDimSize, 0L),
                                  ),
                                )
                                newDimSize
                              }
                            }
                          }
                          newDimSizeI
                        },
                        cb,
                      )(x => IEmitCode(cb, false, x))
                    }
                  }
                  val slicingValueTriples = slicingValueTriplesBuilder.result()

                  outputShape.map(cb) { outputShapeSeq =>
                    new NDArrayProducer() {
                      override def elementType: PType = childProducer.elementType

                      override val shape: IndexedSeq[Value[Long]] = outputShapeSeq

                      override val initAll: EmitCodeBuilder => Unit = cb => {
                        childProducer.initAll(cb)
                        // Need to get the indexingIndices to the right starting points
                        indexingIndices.zipWithIndex.foreach { case (childIdx, ordinalIdx) =>
                          childProducer.initAxis(childIdx)
                          childProducer.stepAxis(childIdx)(cb, indexingValues(ordinalIdx))
                        }
                      }

                      override val initAxis: IndexedSeq[EmitCodeBuilder => Unit] =
                        shape.indices.map { idx => (cb: EmitCodeBuilder) =>
                          val whichSlicingAxis = slicingIndices(idx)
                          val slicingValue = slicingValueTriples(idx)
                          childProducer.initAxis(whichSlicingAxis)(cb)
                          childProducer.stepAxis(whichSlicingAxis)(cb, slicingValue._1)
                        }
                      override val stepAxis: IndexedSeq[(EmitCodeBuilder, Value[Long]) => Unit] =
                        shape.indices.map(idx => { (cb: EmitCodeBuilder, outerStep: Value[Long]) =>
                          // SlicingIndices is a map from my coordinates to my child's coordinates.
                          val whichSlicingAxis = slicingIndices(idx)
                          val (_, _, sliceStep) = slicingValueTriples(idx)
                          val innerStep = cb.newLocal[Long](
                            "ndarray_producer_slice_child_step",
                            sliceStep * outerStep,
                          )
                          childProducer.stepAxis(whichSlicingAxis)(cb, innerStep)
                        })

                      override def loadElementAtCurrentAddr(cb: EmitCodeBuilder): SValue =
                        childProducer.loadElementAtCurrentAddr(cb)
                    }
                  }
                }
              }
            }
          case NDArrayFilter(child, filters) =>
            deforestRecur(child, cb).map(cb) { childProducer: NDArrayProducer =>
              val filterWasMissing = (0 until filters.size).map(i =>
                cb.newField[Boolean](s"ndarray_filter_${i}_was_missing")
              )
              val filtPValues = new Array[SIndexableValue](filters.size)
              val outputShape = childProducer.shape.indices.map(idx =>
                cb.newField[Long](s"ndarray_filter_output_shapes_$idx")
              )

              filters.zipWithIndex.foreach { case (filt, i) =>
                /* Each filt is a sequence that may be missing with elements that may not be
                 * missing. */
                emitI(filt, cb).consume(
                  cb, {
                    cb.assign(outputShape(i), childProducer.shape(i))
                    cb.assign(filterWasMissing(i), true)
                  },
                  {
                    case filtArrayPValue: SIndexableValue =>
                      filtPValues(i) = filtArrayPValue
                      cb.assign(outputShape(i), filtArrayPValue.loadLength().toL)
                      cb.assign(filterWasMissing(i), false)
                  },
                )
              }

              new NDArrayProducer {
                override def elementType: PType = childProducer.elementType

                override val shape: IndexedSeq[Value[Long]] = outputShape

                /* Plan: Keep track of current indices on each axis, use them to step through
                 * filtered */
                // dimensions accordingly.
                val idxVars =
                  shape.indices.map(idx => cb.newLocal[Long](s"ndarray_producer_filter_index_$idx"))

                override val initAll: EmitCodeBuilder => Unit = cb => {
                  idxVars.foreach(idxVar => cb.assign(idxVar, 0L))
                  childProducer.initAll(cb)
                }
                override val initAxis: IndexedSeq[EmitCodeBuilder => Unit] =
                  shape.indices.map { idx => (cb: EmitCodeBuilder) =>
                    {
                      cb.assign(idxVars(idx), 0L)
                      childProducer.initAxis(idx)(cb)
                      cb.if_(
                        filterWasMissing(idx), {
                          /* pass */
                        }, {
                          val startPoint = cb.newLocal[Long](
                            "ndarray_producer_filter_init_axis",
                            filtPValues(idx).loadElement(cb, idxVars(idx).toI).get(
                              cb,
                              s"NDArrayFilter: can't filter on missing index (axis=$idx)",
                            ).asLong.value,
                          )
                          childProducer.stepAxis(idx)(cb, startPoint)
                        },
                      )
                    }
                  }
                override val stepAxis: IndexedSeq[(EmitCodeBuilder, Value[Long]) => Unit] =
                  shape.indices.map { idx => (cb: EmitCodeBuilder, step: Value[Long]) =>
                    {
                      cb.if_(
                        filterWasMissing(idx), {
                          childProducer.stepAxis(idx)(cb, step)
                          cb.assign(idxVars(idx), idxVars(idx) + step)
                        }, {
                          val currentPos = filtPValues(idx).loadElement(cb, idxVars(idx).toI).get(
                            cb,
                            s"NDArrayFilter: can't filter on missing index (axis=$idx)",
                          ).asLong.value
                          cb.assign(idxVars(idx), idxVars(idx) + step)
                          val newPos = filtPValues(idx).loadElement(cb, idxVars(idx).toI).get(
                            cb,
                            s"NDArrayFilter: can't filter on missing index (axis=$idx)",
                          ).asLong.value
                          val stepSize = cb.newLocal[Long](
                            "ndarray_producer_filter_step_size",
                            newPos - currentPos,
                          )
                          childProducer.stepAxis(idx)(cb, stepSize)
                        },
                      )
                    }
                  }

                override def loadElementAtCurrentAddr(cb: EmitCodeBuilder): SValue =
                  childProducer.loadElementAtCurrentAddr(cb)
              }
            }
          case NDArrayAgg(child, axesToSumOut) =>
            deforestRecur(child, cb).map(cb) { childProducer: NDArrayProducer =>
              val childDims = child.typ.asInstanceOf[TNDArray].nDims
              val axesToKeep = (0 until childDims).filter(axis => !axesToSumOut.contains(axis))
              val newOutputShape = axesToKeep.map(idx => childProducer.shape(idx))
              val newOutputShapeComplement = axesToSumOut.map(idx => childProducer.shape(idx))

              val newElementType: PType = child.typ.asInstanceOf[TNDArray].elementType match {
                case TInt32 => PInt32Required
                case TInt64 => PInt64Required
                case TFloat32 => PFloat32Required
                case TFloat64 => PFloat64Required
              }
              new NDArrayProducer {
                override def elementType: PType = newElementType

                override val shape: IndexedSeq[Value[Long]] = newOutputShape

                override val initAll: EmitCodeBuilder => Unit = childProducer.initAll
                /* Important part here is that NDArrayAgg has less axes then its child. We need to
                 * map */
                // between them.
                override val initAxis: IndexedSeq[EmitCodeBuilder => Unit] =
                  axesToKeep.map(idx => childProducer.initAxis(idx))

                override val stepAxis: IndexedSeq[(EmitCodeBuilder, Value[Long]) => Unit] =
                  axesToKeep.map(idx => childProducer.stepAxis(idx))

                override def loadElementAtCurrentAddr(cb: EmitCodeBuilder): SValue = {
                  /* Idea: For each axis that is being summed over, step through and keep a running
                   * sum. */
                  val numericElementType = elementType.asInstanceOf[PNumeric]
                  val runningSum = NumericPrimitives.newLocal(
                    cb,
                    "ndarray_agg_running_sum",
                    numericElementType.virtualType,
                  )
                  cb.assign(runningSum, numericElementType.zero)

                  val initsToSumOut = axesToSumOut.map(idx => childProducer.initAxis(idx))
                  val stepsToSumOut = axesToSumOut.map(idx =>
                    (cb: EmitCodeBuilder) => childProducer.stepAxis(idx)(cb, 1L)
                  )

                  SNDArray.forEachIndexWithInitAndIncColMajor(cb, newOutputShapeComplement,
                    initsToSumOut, stepsToSumOut, "ndarray_producer_ndarray_agg") { (cb, _) =>
                    cb.assign(
                      runningSum,
                      numericElementType.add(
                        runningSum,
                        SType.extractPrimValue(cb, childProducer.loadElementAtCurrentAddr(cb)),
                      ),
                    )
                  }
                  primitive(numericElementType.virtualType, runningSum)
                }
              }
            }
          case _ =>
            val ndI = emitI(x, cb)
            ndI.map(cb)(ndPv => fromSValue(ndPv.asNDArray, cb))
        }
      }

      deforestRecur(x)
    }

    emitNDInSeparateMethod("foo", cb, ndIR, region, env, container, loopEnv)
  }

  def fromSValue(ndSv: SNDArrayValue, cb: EmitCodeBuilder): NDArrayProducer = {
    val ndSvShape = ndSv.shapes
    val strides = ndSv.strides

    fromShapeStridesFirstAddress(
      ndSv.st.elementPType,
      ndSvShape,
      strides,
      ndSv.firstDataAddress,
      cb,
    )
  }

  def fromShapeStridesFirstAddress(
    newElementType: PType,
    ndSvShape: IndexedSeq[Value[Long]],
    strides: IndexedSeq[Value[Long]],
    firstDataAddress: Value[Long],
    cb: EmitCodeBuilder,
  ): NDArrayProducer = {
    val counters =
      ndSvShape.indices.map(i => cb.newLocal[Long](s"ndarray_producer_fall_through_idx_$i"))

    assert(
      ndSvShape.size == strides.size,
      s"shape.size = ${ndSvShape.size} != strides.size = ${strides.size}",
    )

    new NDArrayProducer {
      override def elementType: PType = newElementType
      override val shape: IndexedSeq[Value[Long]] = ndSvShape

      override val initAll: EmitCodeBuilder => Unit = cb =>
        counters.foreach(ctr => cb.assign(ctr, 0L))
      override val initAxis: IndexedSeq[EmitCodeBuilder => Unit] =
        shape.indices.map(i =>
          (cb: EmitCodeBuilder) =>
            cb.assign(counters(i), 0L)
        )
      override val stepAxis: IndexedSeq[(EmitCodeBuilder, Value[Long]) => Unit] =
        shape.indices.map { i => (cb: EmitCodeBuilder, step: Value[Long]) =>
          cb.assign(counters(i), counters(i) + step * strides(i))
        }

      override def loadElementAtCurrentAddr(cb: EmitCodeBuilder): SValue = {
        val offset = counters.foldLeft[Code[Long]](const(0L))((a, b) => a + b)
        val loaded = elementType.loadCheapSCode(cb, firstDataAddress + offset)
        loaded
      }
    }
  }

  def createBroadcastMask(cb: EmitCodeBuilder, shape: IndexedSeq[Value[Long]])
    : IndexedSeq[Value[Long]] = {
    val ffff = 0xffffffffffffffffL
    shape.indices.map { idx =>
      cb.newLocal[Long](s"ndarray_producer_broadcast_mask_$idx", (shape(idx) ceq 1L).mux(0L, ffff))
    }
  }

  def broadcast(cb: EmitCodeBuilder, prod: NDArrayProducer, ctx: String): NDArrayProducer = {
    val broadcastMask = createBroadcastMask(cb, prod.shape)
    val newSteps = prod.stepAxis.indices.map { idx => (cb: EmitCodeBuilder, step: Value[Long]) =>
      {
        val maskedStep =
          cb.newLocal[Long]("ndarray_producer_masked_step", step & broadcastMask(idx))
        prod.stepAxis(idx)(cb, maskedStep)
      }
    }
    prod.copy(astepAxis = newSteps)
  }
}
