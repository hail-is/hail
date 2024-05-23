package is.hail.expr.ir

import is.hail.annotations.Region
import is.hail.asm4s.{const, Code, Value, _}
import is.hail.expr.ir.functions.MathFunctions
import is.hail.io.{AbstractTypedCodecSpec, OutputBuffer}
import is.hail.types.physical._
import is.hail.types.physical.stypes.{EmitType, SValue}
import is.hail.types.physical.stypes.concrete._
import is.hail.types.physical.stypes.interfaces.{
  primitive, SBaseStruct, SIndexableValue, SStreamValue,
}
import is.hail.types.physical.stypes.primitives.{SBooleanValue, SInt32, SInt32Value}
import is.hail.types.virtual.TBaseStruct
import is.hail.utils._

object EmitStreamDistribute {

  def emit(
    cb: EmitCodeBuilder,
    region: Value[Region],
    requestedSplittersAndEndsVal: SIndexableValue,
    childStream: SStreamValue,
    pathVal: SValue,
    comparisonOp: ComparisonOp[_],
    spec: AbstractTypedCodecSpec,
  ): SIndexableValue = {
    val mb = cb.emb
    val pivotsPType = requestedSplittersAndEndsVal.st.storageType().asInstanceOf[PCanonicalArray]
    val requestedSplittersVal = requestedSplittersAndEndsVal.sliceArray(
      cb,
      region,
      pivotsPType,
      1,
      requestedSplittersAndEndsVal.loadLength() - 1,
    )

    val keyType = requestedSplittersVal.st.elementType.asInstanceOf[SBaseStruct]
    val keyPType = pivotsPType.elementType
    val keyFieldNames = keyType.virtualType.fields.map(_.name)

    def compare(cb: EmitCodeBuilder, lelt: EmitValue, relt: EmitValue): Code[Int] = {
      val lhs = lelt.map(cb)(_.asBaseStruct.subset(keyFieldNames: _*))
      val rhs = relt.map(cb)(_.asBaseStruct.subset(keyFieldNames: _*))
      val codeOrdering = comparisonOp.codeOrdering(
        cb.emb.ecb,
        lhs.st.asInstanceOf[SBaseStruct],
        rhs.st.asInstanceOf[SBaseStruct],
      )
      codeOrdering(cb, lhs, rhs).asInstanceOf[Value[Int]]
    }

    def equal(cb: EmitCodeBuilder, lelt: EmitValue, relt: EmitValue): Code[Boolean] =
      compare(cb, lelt, relt) ceq 0

    def lessThan(cb: EmitCodeBuilder, lelt: EmitValue, relt: EmitValue): Code[Boolean] =
      compare(cb, lelt, relt) < 0

    val filledInTreeSize = Code.invokeScalaObject1[Int, Int](
      MathFunctions.getClass,
      "roundToNextPowerOf2",
      requestedSplittersVal.loadLength() + 1,
    )
    val treeHeight = cb.memoize[Int](Code.invokeScalaObject1[Int, Int](
      MathFunctions.getClass,
      "log2",
      filledInTreeSize,
    ))

    val paddedSplittersSize = cb.memoize[Int](const(1) << treeHeight)

    val uniqueSplittersIdx = cb.newLocal[Int]("unique_splitters_idx", 0)

    def cleanupSplitters() = {
      /* Copy each unique splitter into array. If it is seen twice, set a boolean in a parallel
       * array for that */
      // splitter, so we know what identity buckets to make later
      val paddedSplittersPType = PCanonicalArray(keyPType)
      val splittersWasDuplicatedPType = PCanonicalArray(PBooleanRequired)

      val paddedSplittersAddr =
        cb.memoize[Long](paddedSplittersPType.allocate(region, paddedSplittersSize))
      paddedSplittersPType.stagedInitialize(cb, paddedSplittersAddr, paddedSplittersSize)

      val splittersWasDuplicatedLength = paddedSplittersSize
      val splittersWasDuplicatedAddr =
        cb.memoize[Long](splittersWasDuplicatedPType.allocate(region, splittersWasDuplicatedLength))
      splittersWasDuplicatedPType.stagedInitialize(
        cb,
        splittersWasDuplicatedAddr,
        splittersWasDuplicatedLength,
      )
      val splitters: SIndexableValue = paddedSplittersPType.loadCheapSCode(cb, paddedSplittersAddr)

      val requestedSplittersIdx = cb.newLocal[Int]("stream_distribute_splitters_index")
      val lastKeySeen = cb.emb.newEmitLocal("stream_distribute_last_seen", keyType, false)

      cb.for_(
        cb.assign(requestedSplittersIdx, 0),
        requestedSplittersIdx < requestedSplittersVal.loadLength(),
        cb.assign(requestedSplittersIdx, requestedSplittersIdx + 1), {
          val currentSplitter = requestedSplittersVal.loadElement(
            cb,
            requestedSplittersIdx,
          ).memoize(cb, "stream_distribute_current_splitter")
          cb.if_(
            requestedSplittersIdx ceq 0, {
              paddedSplittersPType.elementType.storeAtAddress(
                cb,
                paddedSplittersPType.loadElement(paddedSplittersAddr, paddedSplittersSize, 0),
                region,
                currentSplitter.get(cb),
                false,
              )
              splittersWasDuplicatedPType.elementType.storeAtAddress(
                cb,
                splittersWasDuplicatedPType.loadElement(
                  splittersWasDuplicatedAddr,
                  splittersWasDuplicatedLength,
                  uniqueSplittersIdx,
                ),
                region,
                new SBooleanValue(false),
                false,
              )
              cb.assign(uniqueSplittersIdx, uniqueSplittersIdx + 1)
            }, {
              cb.if_(
                !equal(cb, lastKeySeen, currentSplitter), {
                  // write to pos in splitters
                  paddedSplittersPType.elementType.storeAtAddress(
                    cb,
                    paddedSplittersPType.loadElement(
                      paddedSplittersAddr,
                      paddedSplittersSize,
                      uniqueSplittersIdx,
                    ),
                    region,
                    currentSplitter.get(cb),
                    false,
                  )
                  splittersWasDuplicatedPType.elementType.storeAtAddress(
                    cb,
                    splittersWasDuplicatedPType.loadElement(
                      splittersWasDuplicatedAddr,
                      splittersWasDuplicatedLength,
                      uniqueSplittersIdx,
                    ),
                    region,
                    new SBooleanValue(false),
                    false,
                  )
                  cb.assign(uniqueSplittersIdx, uniqueSplittersIdx + 1)
                },
                splittersWasDuplicatedPType.elementType.storeAtAddress(
                  cb,
                  splittersWasDuplicatedPType.loadElement(
                    splittersWasDuplicatedAddr,
                    splittersWasDuplicatedLength,
                    uniqueSplittersIdx - 1,
                  ),
                  region,
                  new SBooleanValue(true),
                  false,
                ),
              )
            },
          )
          cb.assign(lastKeySeen, currentSplitter)
        },
      )

      val numUniqueSplitters = cb.memoize[Int](uniqueSplittersIdx)

      // Pad out the rest of the splitters array so tree later is balanced.
      cb.for_(
        {},
        uniqueSplittersIdx < paddedSplittersSize,
        cb.assign(uniqueSplittersIdx, uniqueSplittersIdx + 1), {
          cb.if_(
            lastKeySeen.get(cb).asInstanceOf[SBaseStructPointerSettable].a ceq const(0L),
            cb._fatal("paddedSplitterSize was ", paddedSplittersSize.toS),
          )
          val loaded = paddedSplittersPType.loadElement(
            paddedSplittersAddr,
            paddedSplittersSize,
            uniqueSplittersIdx,
          )
          paddedSplittersPType.elementType.storeAtAddress(
            cb,
            loaded,
            region,
            lastKeySeen.get(cb),
            false,
          )
        },
      )

      val splitterWasDuplicated =
        splittersWasDuplicatedPType.loadCheapSCode(cb, splittersWasDuplicatedAddr)
      (splitters, numUniqueSplitters, splitterWasDuplicated)
    }

    def buildTree(paddedSplitters: SIndexableValue, treePType: PCanonicalArray): SIndexableValue = {
      val treeAddr = cb.memoize[Long](treePType.allocate(region, paddedSplittersSize))
      treePType.stagedInitialize(cb, treeAddr, paddedSplittersSize)

      /* Walk through the array one level of the tree at a time, filling in the tree as you go to
       * get a breadth first traversal of the tree. */
      val currentHeight = cb.newLocal[Int]("stream_dist_current_height")
      val treeFillingIndex = cb.newLocal[Int]("stream_dist_tree_filling_idx", 1)
      cb.for_(
        cb.assign(currentHeight, treeHeight - 1),
        currentHeight >= 0,
        cb.assign(currentHeight, currentHeight - 1), {
          val startingPoint = cb.memoize[Int]((const(1) << currentHeight) - 1)
          val inner = cb.newLocal[Int]("stream_dist_tree_inner")
          cb.for_(
            cb.assign(inner, 0),
            inner < (const(1) << (treeHeight - 1 - currentHeight)),
            cb.assign(inner, inner + 1), {
              val elementLoaded = paddedSplitters.loadElement(
                cb,
                startingPoint + inner * (const(1) << (currentHeight + 1)),
              ).get(cb)
              keyPType.storeAtAddress(
                cb,
                treePType.loadElement(treeAddr, treeFillingIndex),
                region,
                elementLoaded,
                false,
              )
              cb.assign(treeFillingIndex, treeFillingIndex + 1)
            },
          )
        },
      )
      // 0th element is garbage, tree elements start at idx 1.
      treePType.loadCheapSCode(cb, treeAddr)
    }

    def createFileMapping(
      numFilesToWrite: Value[Int],
      splitterWasDuplicated: SIndexableValue,
      numberOfBuckets: Value[Int],
      shouldUseIdentityBuckets: Value[Boolean],
    ): SIndexablePointerValue = {
      /* The element classifying algorithm acts as though there are identity buckets for every
       * splitter. We only use identity buckets for elements that repeat */
      /* in splitters list. Since We don't want many empty files, we need to make an array mapping
       * output buckets to files. */
      val fileMappingType = PCanonicalArray(PInt32Required)
      val fileMappingAddr = cb.memoize(fileMappingType.allocate(region, numberOfBuckets))
      fileMappingType.stagedInitialize(cb, fileMappingAddr, numberOfBuckets)

      val bucketIdx = cb.newLocal[Int]("stream_dist_bucket_idx")
      val currentFileToMapTo = cb.newLocal[Int]("stream_dist_mapping_cur_storage", 0)
      def destFileSCode(cb: EmitCodeBuilder) =
        new SInt32Value(cb.memoize((currentFileToMapTo >= numFilesToWrite).mux(
          numFilesToWrite - 1,
          currentFileToMapTo,
        )))

      val indexIncrement = cb.newLocal[Int]("stream_dist_create_file_mapping_increment")
      cb.if_(shouldUseIdentityBuckets, cb.assign(indexIncrement, 2), cb.assign(indexIncrement, 1))

      cb.for_(
        cb.assign(bucketIdx, 0),
        bucketIdx < numberOfBuckets,
        cb.assign(bucketIdx, bucketIdx + indexIncrement), {
          fileMappingType.elementType.storeAtAddress(
            cb,
            fileMappingType.loadElement(fileMappingAddr, numberOfBuckets, bucketIdx),
            region,
            destFileSCode(cb),
            false,
          )
          cb.if_(
            shouldUseIdentityBuckets, {
              cb.assign(
                currentFileToMapTo,
                currentFileToMapTo + splitterWasDuplicated.loadElement(cb, bucketIdx / 2).get(
                  cb
                ).asBoolean.value.toI,
              )
              fileMappingType.elementType.storeAtAddress(
                cb,
                fileMappingType.loadElement(fileMappingAddr, numberOfBuckets, bucketIdx + 1),
                region,
                destFileSCode(cb),
                false,
              )
            },
          )
          cb.assign(currentFileToMapTo, currentFileToMapTo + 1)
        },
      )

      fileMappingType.loadCheapSCode(cb, fileMappingAddr)
    }

    val (paddedSplitters, numUniqueSplitters, splitterWasDuplicated) = cleanupSplitters()
    val tree = buildTree(paddedSplitters, PCanonicalArray(keyPType))

    val shouldUseIdentityBuckets =
      cb.memoize[Boolean](numUniqueSplitters < requestedSplittersVal.loadLength())
    val numberOfBuckets = cb.newLocal[Int]("stream_dist_number_of_buckets")
    cb.if_(
      shouldUseIdentityBuckets,
      cb.assign(numberOfBuckets, const(1) << (treeHeight + 1)),
      cb.assign(numberOfBuckets, const(1) << treeHeight),
    )

    /* Without identity buckets you'd have numUniqueSplitters + 1 buckets, but we have to add an
     * extra for each identity bucket. */
    // FIXME: We should have less files if we aren't writing endpoint buckets.
    val numFilesToWrite = cb.newLocal[Int]("stream_dist_num_files_to_write", 1)
    cb.for_(
      cb.assign(uniqueSplittersIdx, 0),
      uniqueSplittersIdx < numUniqueSplitters,
      cb.assign(uniqueSplittersIdx, uniqueSplittersIdx + 1),
      cb.assign(
        numFilesToWrite,
        numFilesToWrite + 1 + splitterWasDuplicated.loadElement(cb, uniqueSplittersIdx).get(
          cb
        ).asBoolean.value.toI,
      ),
    )

    val fileMapping = createFileMapping(
      numFilesToWrite,
      splitterWasDuplicated,
      numberOfBuckets,
      shouldUseIdentityBuckets,
    )

    val outputBuffers =
      cb.memoize[Array[OutputBuffer]](Code.newArray[OutputBuffer](numFilesToWrite))
    val numElementsPerFile = cb.memoize[Array[Int]](Code.newArray[Int](numFilesToWrite))
    val numBytesPerFile = cb.memoize[Array[Long]](Code.newArray[Long](numFilesToWrite))

    val fileArrayIdx = cb.newLocal[Int]("stream_dist_file_array_idx")

    def makeFileName(cb: EmitCodeBuilder, fileIdx: Value[Int]): Value[String] =
      cb.memoize(pathVal.asString.loadString(cb) concat const("/sorted_part_") concat fileIdx.toS)

    cb.for_(
      cb.assign(fileArrayIdx, 0),
      fileArrayIdx < numFilesToWrite,
      cb.assign(fileArrayIdx, fileArrayIdx + 1), {
        val fileName = makeFileName(cb, fileArrayIdx)
        val ob = cb.memoize(spec.buildCodeOutputBuffer(mb.createUnbuffered(fileName)))
        cb += outputBuffers.update(fileArrayIdx, ob)
        cb += numElementsPerFile.update(fileArrayIdx, 0)
        cb += numBytesPerFile.update(fileArrayIdx, 0)
      },
    )
    /* The element classifying algorithm acts as though there are identity buckets for every
     * splitter. We only use identity buckets for elements that repeat */
    /* in splitters list. Since We don't want many empty files, we need to make an array mapping
     * output buckets to files. */

    val encoder = spec.encodedType.buildEncoder(childStream.st.elementType, cb.emb.ecb)
    val producer = childStream.getProducer(mb)
    producer.memoryManagedConsume(region, cb) { cb =>
      val b = cb.newLocal[Int]("stream_dist_b_i", 1)
      val current = mb.newEmitField("stream_dist_current", childStream.st.elementEmitType)
      cb.assign(current, producer.element)

      val r = cb.newLocal[Int]("stream_dist_r")
      cb.for_(
        cb.assign(r, 0),
        r < treeHeight,
        cb.assign(r, r + 1), {
          val treeAtB = tree.loadElement(cb, b).memoize(cb, "stream_dist_tree_b")
          cb.assign(b, const(2) * b + lessThan(cb, treeAtB, current).toI)
        },
      )
      cb.if_(
        shouldUseIdentityBuckets,
        cb.assign(
          b,
          const(2) * b + 1 - lessThan(
            cb,
            current,
            paddedSplitters.loadElement(cb, b - numberOfBuckets / 2).memoize(
              cb,
              "stream_dist_splitter_compare",
            ),
          ).toI,
        ),
      )

      val fileToUse =
        cb.memoize[Int](fileMapping.loadElement(cb, b - numberOfBuckets).get(cb).asInt.value)

      val ob = cb.memoize[OutputBuffer](outputBuffers(fileToUse))

      cb += ob.writeByte(1.asInstanceOf[Byte])
      val curSV = current.get(cb)
      encoder(cb, curSV, ob)
      cb += numElementsPerFile.update(fileToUse, numElementsPerFile(fileToUse) + 1)
      cb += numBytesPerFile.update(
        fileToUse,
        numBytesPerFile(fileToUse) + curSV.sizeToStoreInBytes(cb).value,
      )
    }

    cb.for_(
      cb.assign(fileArrayIdx, 0),
      fileArrayIdx < numFilesToWrite,
      cb.assign(fileArrayIdx, fileArrayIdx + 1), {
        val ob = cb.memoize[OutputBuffer](outputBuffers(fileArrayIdx))
        cb += ob.writeByte(0.asInstanceOf[Byte])
        cb += ob.invoke[Unit]("close")
      },
    )

    val intervalType = PCanonicalInterval(keyPType.setRequired(false), true)
    val returnType = PCanonicalArray(
      PCanonicalStruct(
        ("interval", intervalType),
        ("fileName", PCanonicalStringRequired),
        ("numElements", PInt32Required),
        ("numBytes", PInt64Required),
      ),
      true,
    )

    val min = requestedSplittersAndEndsVal.loadElement(cb, 0).memoize(cb, "stream_dist_min")
    val firstSplitter = paddedSplitters.loadElement(cb, 0).memoize(cb, "stream_dist_first_splitter")
    val max = requestedSplittersAndEndsVal.loadElement(
      cb,
      requestedSplittersAndEndsVal.loadLength() - 1,
    ).memoize(cb, "stream_dist_min")
    val lastSplitter = paddedSplitters.loadElement(cb, paddedSplitters.loadLength() - 1).memoize(
      cb,
      "stream_dist_last_splitter",
    )

    val skipMinInterval = cb.memoize(equal(
      cb,
      min,
      firstSplitter,
    ) && splitterWasDuplicated.loadElement(cb, 0).get(cb).asBoolean.value)
    val skipMaxInterval = cb.memoize(equal(cb, max, lastSplitter))

    val (pushElement, finisher) = returnType.constructFromFunctions(
      cb,
      region,
      cb.memoize(numFilesToWrite - skipMinInterval.toI - skipMaxInterval.toI),
      false,
    )

    val stackStructType = new SStackStruct(
      returnType.virtualType.elementType.asInstanceOf[TBaseStruct],
      IndexedSeq(
        EmitType(intervalType.sType, true),
        EmitType(SJavaString, true),
        EmitType(SInt32, true),
      ),
    )

    // Add first, but only if min != first key.
    cb.if_(
      !skipMinInterval, {
        val firstInterval = intervalType.constructFromCodes(
          cb,
          region,
          min,
          firstSplitter,
          true,
          cb.memoize(!splitterWasDuplicated.loadElement(cb, 0).get(cb).asBoolean.value),
        )

        pushElement(
          cb,
          IEmitCode.present(
            cb,
            new SStackStructValue(
              stackStructType,
              IndexedSeq(
                EmitValue.present(firstInterval),
                EmitValue.present(SJavaString.construct(cb, makeFileName(cb, 0))),
                EmitValue.present(primitive(cb.memoize(numElementsPerFile(0)))),
                EmitValue.present(primitive(cb.memoize(numBytesPerFile(0)))),
              ),
            ),
          ),
        )
      },
    )

    cb.for_(
      { cb.assign(uniqueSplittersIdx, 0); cb.assign(fileArrayIdx, 1) },
      uniqueSplittersIdx < numUniqueSplitters,
      cb.assign(uniqueSplittersIdx, uniqueSplittersIdx + 1), {
        cb.if_(
          uniqueSplittersIdx cne 0, {
            val intervalFromLastToThis = intervalType.constructFromCodes(
              cb,
              region,
              EmitCode.fromI(cb.emb)(cb => paddedSplitters.loadElement(cb, uniqueSplittersIdx - 1)),
              EmitCode.fromI(cb.emb)(cb => paddedSplitters.loadElement(cb, uniqueSplittersIdx)),
              false,
              cb.memoize(
                !splitterWasDuplicated.loadElement(cb, uniqueSplittersIdx).get(cb).asBoolean.value
              ),
            )

            pushElement(
              cb,
              IEmitCode.present(
                cb,
                new SStackStructValue(
                  stackStructType,
                  IndexedSeq(
                    EmitValue.present(intervalFromLastToThis),
                    EmitValue.present(SJavaString.construct(cb, makeFileName(cb, fileArrayIdx))),
                    EmitValue.present(primitive(cb.memoize(numElementsPerFile(fileArrayIdx)))),
                    EmitValue.present(primitive(cb.memoize(numBytesPerFile(fileArrayIdx)))),
                  ),
                ),
              ),
            )

            cb.assign(fileArrayIdx, fileArrayIdx + 1)
          },
        )

        // Now, maybe have to make an identity bucket.
        cb.if_(
          splitterWasDuplicated.loadElement(cb, uniqueSplittersIdx).get(cb).asBoolean.value, {
            val identityInterval = intervalType.constructFromCodes(
              cb,
              region,
              EmitCode.fromI(cb.emb)(cb => paddedSplitters.loadElement(cb, uniqueSplittersIdx)),
              EmitCode.fromI(cb.emb)(cb => paddedSplitters.loadElement(cb, uniqueSplittersIdx)),
              true,
              true,
            )

            pushElement(
              cb,
              IEmitCode.present(
                cb,
                new SStackStructValue(
                  stackStructType,
                  IndexedSeq(
                    EmitValue.present(identityInterval),
                    EmitValue.present(SJavaString.construct(cb, makeFileName(cb, fileArrayIdx))),
                    EmitValue.present(primitive(cb.memoize(numElementsPerFile(fileArrayIdx)))),
                    EmitValue.present(primitive(cb.memoize(numBytesPerFile(fileArrayIdx)))),
                  ),
                ),
              ),
            )

            cb.assign(fileArrayIdx, fileArrayIdx + 1)
          },
        )
      },
    )

    // Add last, but only if max != last key
    cb.if_(
      !skipMaxInterval, {
        val lastInterval = intervalType.constructFromCodes(
          cb,
          region,
          EmitCode.fromI(cb.emb)(cb => paddedSplitters.loadElement(cb, uniqueSplittersIdx - 1)),
          EmitCode.fromI(cb.emb)(cb =>
            requestedSplittersAndEndsVal.loadElement(
              cb,
              requestedSplittersAndEndsVal.loadLength() - 1,
            )
          ),
          false,
          true,
        )

        pushElement(
          cb,
          IEmitCode.present(
            cb,
            new SStackStructValue(
              stackStructType,
              IndexedSeq(
                EmitValue.present(lastInterval),
                EmitValue.present(SJavaString.construct(cb, makeFileName(cb, fileArrayIdx))),
                EmitValue.present(primitive(cb.memoize(numElementsPerFile(fileArrayIdx)))),
                EmitValue.present(primitive(cb.memoize(numBytesPerFile(fileArrayIdx)))),
              ),
            ),
          ),
        )
      },
    )

    finisher(cb)
  }

}
