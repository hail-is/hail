package is.hail.expr.ir

import is.hail.annotations.Region
import is.hail.asm4s.{Code, Value, const}
import is.hail.expr.ir.functions.MathFunctions
import is.hail.expr.ir.orderings.StructOrdering
import is.hail.expr.ir.streams.EmitStream
import is.hail.io.{AbstractTypedCodecSpec, OutputBuffer}
import is.hail.types.VirtualTypeWithReq
import is.hail.types.physical.{PBooleanRequired, PCanonicalArray, PCanonicalInterval, PCanonicalStringRequired, PCanonicalStruct, PContainer, PInt32Required}
import is.hail.types.physical.stypes.{EmitType, SValue}
import is.hail.types.physical.stypes.concrete.{SIndexablePointer, SIndexablePointerCode, SJavaString, SStackStruct, SStackStructCode}
import is.hail.types.physical.stypes.interfaces.{SBaseStruct, SIndexableCode, SIndexableValue, SStreamCode, SStringCode, primitive}
import is.hail.types.physical.stypes.primitives.{SBooleanCode, SInt32, SInt32Code}
import is.hail.types.virtual.{TArray, TBaseStruct, TStruct}
import is.hail.utils._
import is.hail.asm4s._

object EmitStreamDistribute {

  def emit(cb: EmitCodeBuilder, region: Value[Region], requestedSplittersAndEndsVal: SIndexableValue, childStream: SStreamCode, pathVal: SValue, spec: AbstractTypedCodecSpec): SIndexableCode = {
    val mb = cb.emb
    val pivotsPType = requestedSplittersAndEndsVal.st.storageType().asInstanceOf[PCanonicalArray]
    val requestedSplittersVal = requestedSplittersAndEndsVal.sliceArray(cb, region, pivotsPType, 1, requestedSplittersAndEndsVal.loadLength() - 1).memoize(cb, "foo")

    val keyType = requestedSplittersVal.st.elementType.asInstanceOf[SBaseStruct]
    val keyPType = pivotsPType.elementType
    val keyFieldNames = keyType.virtualType.fields.map(_.name)

    def compare(cb: EmitCodeBuilder, lelt: EmitValue, relt: EmitValue): Code[Int] = {
      val lhs = EmitCode.fromI(mb)(cb => lelt.toI(cb).map(cb)(_.asBaseStruct.memoize(cb, "stream_dist_comp_lhs").subset(keyFieldNames: _*).get))
      val rhs = EmitCode.fromI(mb)(cb => relt.toI(cb).map(cb)(_.asBaseStruct.memoize(cb, "stream_dist_comp_rhs").subset(keyFieldNames: _*).get))
      StructOrdering.make(lhs.st.asInstanceOf[SBaseStruct], rhs.st.asInstanceOf[SBaseStruct],
        cb.emb.ecb, missingFieldsEqual = true)
        .compare(cb, lhs, rhs, missingEqual = true)
    }

    def equal(cb: EmitCodeBuilder, lelt: EmitValue, relt: EmitValue): Code[Boolean] = compare(cb, lelt, relt) ceq 0

    def lessThan(cb: EmitCodeBuilder, lelt: EmitValue, relt: EmitValue): Code[Boolean] = compare(cb, lelt, relt) < 0

    val filledInTreeSize = Code.invokeScalaObject1[Int, Int](MathFunctions.getClass, "roundToNextPowerOf2", requestedSplittersVal.loadLength() + 1)
    val treeHeight: Value[Int] = cb.newLocal[Int]("stream_dist_tree_height", Code.invokeScalaObject1[Int, Int](MathFunctions.getClass, "log2", filledInTreeSize))

    val paddedSplittersSize = cb.newLocal[Int]("stream_dist_padded_splitter_size", const(1) << treeHeight)

    val uniqueSplittersIdx = cb.newLocal[Int]("unique_splitters_idx", 0)

    def cleanupSplitters() = {
      // Copy each unique splitter into array. If it is seen twice, set a boolean in a parallel array for that
      // splitter, so we know what identity buckets to make later
      val paddedSplittersPType = PCanonicalArray(keyPType)
      val splittersWasDuplicatedPType = PCanonicalArray(PBooleanRequired)

      val paddedSplittersAddr = cb.newLocal[Long]("stream_dist_splitters_addr", paddedSplittersPType.allocate(region, paddedSplittersSize))
      paddedSplittersPType.stagedInitialize(cb, paddedSplittersAddr, paddedSplittersSize)

      val splittersWasDuplicatedLength = paddedSplittersSize
      val splittersWasDuplicatedAddr = cb.newLocal[Long]("stream_dist_dupe_splitters_addr", splittersWasDuplicatedPType.allocate(region, splittersWasDuplicatedLength))
      splittersWasDuplicatedPType.stagedInitialize(cb, splittersWasDuplicatedAddr, splittersWasDuplicatedLength)
      val splitters: SIndexableValue = new SIndexablePointerCode(SIndexablePointer(paddedSplittersPType), paddedSplittersAddr).memoize(cb, "stream_distribute_splitters_deduplicated") // last element is duplicated, otherwise this is sorted without duplicates.

      val requestedSplittersIdx = cb.newLocal[Int]("stream_distribute_splitters_index")
      val lastKeySeen = cb.emb.newEmitLocal("stream_distribute_last_seen", keyType, false)

      cb.forLoop(cb.assign(requestedSplittersIdx, 0), requestedSplittersIdx < requestedSplittersVal.loadLength(), cb.assign(requestedSplittersIdx, requestedSplittersIdx + 1), {
        val currentSplitter = requestedSplittersVal.loadElement(cb, requestedSplittersIdx).memoize(cb, "stream_distribute_current_splitter")
        cb.ifx(requestedSplittersIdx ceq 0, {
          paddedSplittersPType.elementType.storeAtAddress(cb, paddedSplittersPType.loadElement(paddedSplittersAddr, paddedSplittersSize, 0), region, currentSplitter.get(cb).get, false)
          splittersWasDuplicatedPType.elementType.storeAtAddress(cb, splittersWasDuplicatedPType.loadElement(splittersWasDuplicatedAddr, splittersWasDuplicatedLength, uniqueSplittersIdx), region, new SBooleanCode(false), false)
          cb.assign(uniqueSplittersIdx, uniqueSplittersIdx + 1)
        }, {
          cb.ifx(!equal(cb, lastKeySeen, currentSplitter), {
            // write to pos in splitters
            paddedSplittersPType.elementType.storeAtAddress(cb, paddedSplittersPType.loadElement(paddedSplittersAddr, paddedSplittersSize, uniqueSplittersIdx), region, currentSplitter.get(cb).get, false)
            splittersWasDuplicatedPType.elementType.storeAtAddress(cb, splittersWasDuplicatedPType.loadElement(splittersWasDuplicatedAddr, splittersWasDuplicatedLength, uniqueSplittersIdx), region, new SBooleanCode(false), false)
            cb.assign(uniqueSplittersIdx, uniqueSplittersIdx + 1)
          }, {
            splittersWasDuplicatedPType.elementType.storeAtAddress(cb, splittersWasDuplicatedPType.loadElement(splittersWasDuplicatedAddr, splittersWasDuplicatedLength, uniqueSplittersIdx - 1), region, new SBooleanCode(true), false)
          })
        })
        cb.assign(lastKeySeen, currentSplitter)
      })

      val numUniqueSplitters = cb.newLocal[Int]("stream_distribute_num_unique_splitters", uniqueSplittersIdx)

      // Pad out the rest of the splitters array so tree later is balanced.
      cb.forLoop({}, uniqueSplittersIdx < paddedSplittersSize, cb.assign(uniqueSplittersIdx, uniqueSplittersIdx + 1), {
        paddedSplittersPType.elementType.storeAtAddress(cb, paddedSplittersPType.loadElement(paddedSplittersAddr, paddedSplittersSize, uniqueSplittersIdx), region, lastKeySeen.get(cb).get, false)
      })

      val splitterWasDuplicated = new SIndexablePointerCode(SIndexablePointer(splittersWasDuplicatedPType), splittersWasDuplicatedAddr).memoize(cb, "stream_distrib_was_duplicated") // Same length as splitters, but full of booleans of whether it was initially duplicated.
      (splitters, numUniqueSplitters, splitterWasDuplicated)
    }

    def buildTree(paddedSplitters: SIndexableValue, treePType: PCanonicalArray): SIndexableValue = {
      val treeAddr = cb.newLocal[Long]("stream_dist_tree_addr", treePType.allocate(region, paddedSplittersSize))
      treePType.stagedInitialize(cb, treeAddr, paddedSplittersSize)

      /*
      Walk through the array one level of the tree at a time, filling in the tree as you go to get a breadth
      first traversal of the tree.
     */
      val currentHeight = cb.newLocal[Int]("stream_dist_current_height")
      val treeFillingIndex = cb.newLocal[Int]("stream_dist_tree_filling_idx", 1)
      cb.forLoop(cb.assign(currentHeight, treeHeight - 1), currentHeight >= 0, cb.assign(currentHeight, currentHeight - 1), {
        val startingPoint: Value[Int] = cb.newLocal[Int]("stream_dist_starting_point", (const(1) << currentHeight) - 1)
        val inner = cb.newLocal[Int]("stream_dist_tree_inner")
        cb.forLoop(cb.assign(inner, 0), inner < (const(1) << (treeHeight - 1 - currentHeight)), cb.assign(inner, inner + 1), {
          val elementLoaded = paddedSplitters.loadElement(cb, startingPoint + inner * (const(1) << (currentHeight + 1))).get(cb)
          keyPType.storeAtAddress(cb, treePType.loadElement(treeAddr, treeFillingIndex), region,
            elementLoaded, false)
          cb.assign(treeFillingIndex, treeFillingIndex + 1)
        })
      })
      // 0th element is garbage, tree elements start at idx 1.
      new SIndexablePointerCode(SIndexablePointer(treePType), treeAddr).memoize(cb, "stream_dist_tree")
    }

    def createFileMapping(numFilesToWrite: Value[Int], splitterWasDuplicated: SIndexableValue, numberOfBuckets: Value[Int], shouldUseIdentityBuckets: Value[Boolean]) = {
      // The element classifying algorithm acts as though there are identity buckets for every splitter. We only use identity buckets for elements that repeat
      // in splitters list. Since We don't want many empty files, we need to make an array mapping output buckets to files.
      val fileMappingType = PCanonicalArray(PInt32Required)
      val fileMappingAddr = cb.newLocal("stream_dist_file_map_addr", fileMappingType.allocate(region, numberOfBuckets))
      fileMappingType.stagedInitialize(cb, fileMappingAddr, numberOfBuckets)

      val bucketIdx = cb.newLocal[Int]("stream_dist_bucket_idx")
      val currentFileToMapTo = cb.newLocal[Int]("stream_dist_mapping_cur_storage", 0)
      def destFileSCode() = new SInt32Code((currentFileToMapTo >= numFilesToWrite).mux(numFilesToWrite - 1, currentFileToMapTo))

      val indexIncrement = cb.newLocal[Int]("stream_dist_create_file_mapping_increment")
      cb.ifx(shouldUseIdentityBuckets, cb.assign(indexIncrement, 2), cb.assign(indexIncrement, 1))

      cb.forLoop(cb.assign(bucketIdx, 0), bucketIdx < numberOfBuckets, cb.assign(bucketIdx, bucketIdx + indexIncrement), {
        fileMappingType.elementType.storeAtAddress(cb, fileMappingType.loadElement(fileMappingAddr, numberOfBuckets, bucketIdx), region, destFileSCode(), false)
        cb.ifx(shouldUseIdentityBuckets, {
          cb.assign(currentFileToMapTo, currentFileToMapTo + splitterWasDuplicated.loadElement(cb, bucketIdx / 2).get(cb).asBoolean.boolCode(cb).toI)
          fileMappingType.elementType.storeAtAddress(cb, fileMappingType.loadElement(fileMappingAddr, numberOfBuckets, bucketIdx + 1), region, destFileSCode(), false)
        })
        cb.assign(currentFileToMapTo, currentFileToMapTo + 1)
      })

      new SIndexablePointerCode(SIndexablePointer(fileMappingType), fileMappingAddr).memoize(cb, "stream_dist_file_map")
    }

    val (paddedSplitters, numUniqueSplitters, splitterWasDuplicated) = cleanupSplitters()
    val tree = buildTree(paddedSplitters, PCanonicalArray(keyPType))

    val shouldUseIdentityBuckets = cb.newLocal[Boolean]("stream_dist_use_identity_buckets", numUniqueSplitters < requestedSplittersVal.loadLength())
    val numberOfBuckets = cb.newLocal[Int]("stream_dist_number_of_buckets")
    cb.ifx(shouldUseIdentityBuckets,
      cb.assign(numberOfBuckets, const(1) << (treeHeight + 1)),
      cb.assign(numberOfBuckets, const(1) << treeHeight))

    // Without identity buckets you'd have numUniqueSplitters + 1 buckets, but we have to add an extra for each identity bucket.
    val numFilesToWrite = cb.newLocal[Int]("stream_dist_num_files_to_write", 1)
    cb.forLoop(cb.assign(uniqueSplittersIdx, 0), uniqueSplittersIdx < numUniqueSplitters, cb.assign(uniqueSplittersIdx, uniqueSplittersIdx + 1), {
      cb.assign(numFilesToWrite, numFilesToWrite + 1 + splitterWasDuplicated.loadElement(cb, uniqueSplittersIdx).get(cb).asBoolean.boolCode(cb).toI)
    })

    val fileMapping = createFileMapping(numFilesToWrite, splitterWasDuplicated, numberOfBuckets, shouldUseIdentityBuckets)

    val outputBuffers = cb.newLocal[Array[OutputBuffer]]("stream_dist_output_buffers", Code.newArray[OutputBuffer](numFilesToWrite))
    val numElementsPerFile = cb.newLocal[Array[Int]]("stream_dist_elements_per_file", Code.newArray[Int](numFilesToWrite))
    val fileArrayIdx = cb.newLocal[Int]("stream_dist_file_array_idx")

    def makeFileName(fileIdx: Code[Int]): Code[String] = {
      pathVal.get.asString.loadString() concat const("/sorted_part_") concat (fileIdx.toS)
    }

    cb.forLoop(cb.assign(fileArrayIdx, 0), fileArrayIdx < numFilesToWrite, cb.assign(fileArrayIdx, fileArrayIdx + 1), {
      val ob = cb.newLocal[OutputBuffer]("stream_dist_file_ob")
      val fileName = cb.newLocal[String]("file_to_write", makeFileName(fileArrayIdx))
      cb.assign(ob, spec.buildCodeOutputBuffer(mb.create(fileName)))
      cb += outputBuffers.update(fileArrayIdx, ob)
      cb += numElementsPerFile.update(fileArrayIdx, 0)
    })
    // The element classifying algorithm acts as though there are identity buckets for every splitter. We only use identity buckets for elements that repeat
    // in splitters list. Since We don't want many empty files, we need to make an array mapping output buckets to files.

    val encoder = spec.encodedType.buildEncoder(childStream.st.elementType, cb.emb.ecb)
    childStream.producer.memoryManagedConsume(region, cb) { cb =>
      val b = cb.newLocal[Int]("stream_dist_b_i", 1)
      val current = mb.newEmitField("stream_dist_current", childStream.st.elementEmitType)
      cb.assign(current, childStream.producer.element)

      val r = cb.newLocal[Int]("stream_dist_r")
      cb.forLoop(cb.assign(r, 0), r < treeHeight, cb.assign(r, r + 1), {
        val treeAtB = tree.loadElement(cb, b).memoize(cb, "stream_dist_tree_b")
        cb.assign(b, const(2) * b + lessThan(cb, treeAtB, current).toI)
      })
      cb.ifx(shouldUseIdentityBuckets, {
        cb.assign(b, const(2) * b + 1 - lessThan(cb, current, paddedSplitters.loadElement(cb, b - numberOfBuckets / 2).memoize(cb, "stream_dist_splitter_compare")).toI)
      })

      val fileToUse = cb.newLocal[Int]("stream_dist_index_of_file_to_write", fileMapping.loadElement(cb, b - numberOfBuckets).get(cb).asInt.intCode(cb))

      val ob = cb.newLocal[OutputBuffer]("outputBuffer_to_write", outputBuffers(fileToUse))

      cb += ob.writeByte(1.asInstanceOf[Byte])
      encoder(cb, current.get(cb).get, ob)
      cb += numElementsPerFile.update(fileToUse, numElementsPerFile(fileToUse) + 1)
    }

    cb.forLoop(cb.assign(fileArrayIdx, 0), fileArrayIdx < numFilesToWrite, cb.assign(fileArrayIdx, fileArrayIdx + 1), {
      val ob = cb.newLocal[OutputBuffer]("stream_dist_output_buffer_to_clean_up", outputBuffers(fileArrayIdx))
      cb += ob.writeByte(0.asInstanceOf[Byte])
      cb += ob.invoke[Unit]("close")
    })

    val intervalType = PCanonicalInterval(keyPType.setRequired(false), true)
    val returnType = PCanonicalArray(PCanonicalStruct(("interval", intervalType), ("fileName", PCanonicalStringRequired), ("numElements", PInt32Required)), true)

    val (pushElement, finisher) = returnType.constructFromFunctions(cb, region, numFilesToWrite, false)

    val stackStructType = new SStackStruct(returnType.virtualType.elementType.asInstanceOf[TBaseStruct], IndexedSeq(
      EmitType(intervalType.sType, true),
      EmitType(SJavaString, true),
      EmitType(SInt32, true)
    ))

    // Add first
    val firstInterval = intervalType.constructFromCodes(cb, region,
      EmitCode.fromI(cb.emb)(cb => requestedSplittersAndEndsVal.loadElement(cb, 0)),
      EmitCode.fromI(cb.emb)(cb => paddedSplitters.loadElement(cb, 0)),
      EmitCode.present(cb.emb, primitive(false)),
      EmitCode.present(cb.emb,  new SBooleanCode(!splitterWasDuplicated.loadElement(cb, 0).get(cb).asBoolean.boolCode(cb)))
    )

    pushElement(cb, IEmitCode.present(cb, new SStackStructCode(stackStructType, IndexedSeq(
      EmitCode.present(cb.emb, firstInterval),
      EmitCode.present(cb.emb, SJavaString.construct(makeFileName(0))),
      EmitCode.present(cb.emb, primitive(numElementsPerFile(0)))
    ))))

    cb.forLoop({cb.assign(uniqueSplittersIdx, 0); cb.assign(fileArrayIdx, 1) }, uniqueSplittersIdx < numUniqueSplitters, cb.assign(uniqueSplittersIdx, uniqueSplittersIdx + 1), {
      cb.ifx(uniqueSplittersIdx cne 0, {
        val intervalFromLastToThis = intervalType.constructFromCodes(cb, region,
          EmitCode.fromI(cb.emb)(cb => paddedSplitters.loadElement(cb, uniqueSplittersIdx - 1)),
          EmitCode.fromI(cb.emb)(cb => paddedSplitters.loadElement(cb, uniqueSplittersIdx)),
          EmitCode.present(cb.emb, primitive(false)),
          EmitCode.present(cb.emb, primitive(!splitterWasDuplicated.loadElement(cb, uniqueSplittersIdx).get(cb).asBoolean.boolCode(cb)))
        )

        pushElement(cb, IEmitCode.present(cb, new SStackStructCode(stackStructType, IndexedSeq(
          EmitCode.present(cb.emb, intervalFromLastToThis),
          EmitCode.present(cb.emb, SJavaString.construct(makeFileName(fileArrayIdx))),
          EmitCode.present(cb.emb, primitive(numElementsPerFile(fileArrayIdx)))
        ))))

        cb.assign(fileArrayIdx, fileArrayIdx + 1)
      })

      // Now, maybe have to make an identity bucket.
      cb.ifx(splitterWasDuplicated.loadElement(cb, uniqueSplittersIdx).get(cb).asBoolean.boolCode(cb), {
        val identityInterval = intervalType.constructFromCodes(cb, region,
          EmitCode.fromI(cb.emb)(cb => paddedSplitters.loadElement(cb, uniqueSplittersIdx)),
          EmitCode.fromI(cb.emb)(cb => paddedSplitters.loadElement(cb, uniqueSplittersIdx)),
          EmitCode.present(cb.emb, primitive(true)),
          EmitCode.present(cb.emb, primitive(true))
        )

        pushElement(cb, IEmitCode.present(cb, new SStackStructCode(stackStructType, IndexedSeq(
          EmitCode.present(cb.emb, identityInterval),
          EmitCode.present(cb.emb, SJavaString.construct(makeFileName(fileArrayIdx))),
          EmitCode.present(cb.emb, primitive(numElementsPerFile(fileArrayIdx)))
        ))))

        cb.assign(fileArrayIdx, fileArrayIdx + 1)
      })
    })

    // Add last
    val lastInterval = intervalType.constructFromCodes(cb, region,
      EmitCode.fromI(cb.emb)(cb => paddedSplitters.loadElement(cb, uniqueSplittersIdx - 1)),
      EmitCode.fromI(cb.emb)(cb => requestedSplittersAndEndsVal.loadElement(cb, requestedSplittersAndEndsVal.loadLength() - 1)),
      EmitCode.present(cb.emb, primitive(false)),
      EmitCode.present(cb.emb, primitive(false))
    )

    pushElement(cb, IEmitCode.present(cb, new SStackStructCode(stackStructType, IndexedSeq(
      EmitCode.present(cb.emb, lastInterval),
      EmitCode.present(cb.emb, SJavaString.construct(makeFileName(fileArrayIdx))),
      EmitCode.present(cb.emb,  primitive(numElementsPerFile(fileArrayIdx)))
    ))))

    finisher(cb)
  }

}
