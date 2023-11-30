package is.hail.io.index

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.TaskFinalizer
import is.hail.expr.ir.functions.IntervalFunctions.{arrayOfStructFindIntervalRange, compareStructWithPartitionIntervalEndpoint}
import is.hail.expr.ir.{BinarySearch, EmitCode, EmitCodeBuilder, EmitMethodBuilder, EmitValue, IEmitCode}
import is.hail.io.AbstractTypedCodecSpec
import is.hail.io.fs.FS
import is.hail.types.physical.stypes.concrete._
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.physical.stypes.{SSettable, SValue}
import is.hail.types.physical.{PCanonicalArray, PCanonicalBaseStruct}
import is.hail.types.virtual.{TInt64, TTuple}
import is.hail.utils._

import java.io.{Closeable, InputStream}

case class VariableMetadata(
  branchingFactor: Int,
  height: Int,
  nKeys: Long,
  rootOffset: Long,
  attributes: Map[String, Any]
)

class StagedIndexReader(emb: EmitMethodBuilder[_], leafCodec: AbstractTypedCodecSpec, internalCodec: AbstractTypedCodecSpec) {
  private[this] val cache: Settable[LongToRegionValueCache] = emb.genFieldThisRef[LongToRegionValueCache]("index_cache")
  private[this] val metadata: Settable[VariableMetadata] = emb.genFieldThisRef[VariableMetadata]("index_file_metadata")

  private[this] val is: Settable[ByteTrackingInputStream] = emb.genFieldThisRef[ByteTrackingInputStream]("index_is")

  private[this] val leafPType = leafCodec.encodedType.decodedPType(leafCodec.encodedVirtualType)
  private[this] val internalPType = internalCodec.encodedType.decodedPType(internalCodec.encodedVirtualType)

  private[this] val leafChildType = leafPType.asInstanceOf[PCanonicalBaseStruct].types(1).asInstanceOf[PCanonicalArray].elementType.sType.asInstanceOf[SBaseStruct]
  private[this] val leafChildLocalType = SStackStruct(leafChildType.virtualType, leafChildType.fieldEmitTypes)

  private[this] val queryResultStartIndex: Settable[Long] = emb.genFieldThisRef[Long]("index_resultIndex")
  private[this] val queryResultStartLeaf: SSettable = emb.newPField("index_resultOffset", leafChildLocalType)

  private[this] val leafDec = leafCodec.encodedType.buildDecoder(leafCodec.encodedVirtualType, emb.ecb)
  private[this]val internalDec = internalCodec.encodedType.buildDecoder(internalCodec.encodedVirtualType, emb.ecb)

  def nKeys(cb: EmitCodeBuilder): Value[Long] = cb.memoize(metadata.invoke[Long]("nKeys"))

  def initialize(cb: EmitCodeBuilder,
    indexPath: Value[String]
  ): Unit = {
    val fs = cb.emb.getFS
    cb.assign(cache, Code.newInstance[LongToRegionValueCache, Int](16))
    cb.assign(metadata, Code.invokeScalaObject2[FS, String, IndexMetadataUntypedJSON](
      IndexReader.getClass, "readUntyped", fs, indexPath
    ).invoke[VariableMetadata]("toFileMetadata"))

    // FIXME: hardcoded. Will break if we change spec -- assumption not introduced with this code, but propagated.
    cb.assign(is, Code.newInstance[ByteTrackingInputStream, InputStream](cb.emb.openUnbuffered(indexPath.concat("/index"), false)))

  }

  def addToFinalizer(cb: EmitCodeBuilder, finalizer: Value[TaskFinalizer]): Unit = {
    cb += finalizer.invoke[Closeable, Unit]("addCloseable", cache)
    cb += finalizer.invoke[Closeable, Unit]("addCloseable", is)
  }

  def close(cb: EmitCodeBuilder): Unit = {
    cb += is.invoke[Unit]("close")
    cb += cache.invoke[Unit]("free")
    cb.assign(is, Code._null)
    cb.assign(cache, Code._null)
    cb.assign(metadata, Code._null)
  }

  def queryBound(cb: EmitCodeBuilder,
    endpoint: SBaseStructValue,
    leansRight: Value[Boolean]
  ): SBaseStructValue = {
    val rootLevel = cb.memoize(metadata.invoke[Int]("height") - 1)
    val rootOffset = cb.memoize(metadata.invoke[Long]("rootOffset"))
    val nKeys = this.nKeys(cb)

    val index = cb.newLocal[Long]("queryInterval_startIdx")
    val leaf = cb.newSLocal(leafChildLocalType, "queryInterval_startOffset")

    val LReturn = CodeLabel()

    // handle the cases where the query is less than all keys (including the
    // empty index case), to establish the precondition of runQuery, and as a
    // fast path for a common case
    cb.if_(nKeys.ceq(0), {
      cb.assign(index, 0L)
      cb.goto(LReturn)
    })

    val rootChildren = readInternalNode(cb, rootOffset).loadField(cb, "children").get(cb).asIndexable
    val firstChild = rootChildren.loadElement(cb, 0).get(cb).asBaseStruct
    val firstKey = firstChild.loadField(cb, "first_key").get(cb).asBaseStruct
    val compEndpointWithFirstKey =
      compareStructWithPartitionIntervalEndpoint(cb, firstKey, endpoint, leansRight)
    cb.if_(compEndpointWithFirstKey > 0, {
      cb.assign(index, firstChild.loadField(cb, "first_idx").get(cb).asLong.value)
      cb.assign(leaf, getFirstLeaf(cb, firstChild))
      cb.goto(LReturn)
    })

    queryBound(cb, endpoint, leansRight, rootLevel, rootOffset, nKeys, leaf)
    cb.assign(index, queryResultStartIndex)
    cb.assign(leaf, queryResultStartLeaf)

    cb.define(LReturn)

    SStackStruct.constructFromArgs(cb, null, TTuple(TInt64, leafChildType.virtualType),
      EmitCode.present(cb.emb, primitive(index)),
      EmitCode.fromI(cb.emb)(cb => IEmitCode(cb, index ceq nKeys, leaf)))
  }

  /**
   * returns tuple of (start index, end index, starting leaf)
   * memory of starting leaf is not owned by `region`, consumers should deep copy if necessary
   * starting leaf IS MISSING if (end_idx - start_idx == 0)
   */
  def queryInterval(cb: EmitCodeBuilder, interval: SIntervalValue): SBaseStructValue = {
    val rootLevel = cb.memoize(metadata.invoke[Int]("height") - 1)
    val rootOffset = cb.memoize(metadata.invoke[Long]("rootOffset"))
    val nKeys = this.nKeys(cb)

    val startIdx = cb.newLocal[Long]("queryInterval_startIdx")
    val startLeaf = cb.newSLocal(leafChildLocalType, "queryInterval_startOffset")
    val endIdx = cb.newLocal[Long]("queryInterval_endIdx")

    val startKey = interval.loadStart(cb).get(cb).asBaseStruct
    val startLeansRight = cb.memoize(!interval.includesStart)
    val endKey = interval.loadEnd(cb).get(cb).asBaseStruct
    val endLeansRight = interval.includesEnd

    val LReturn = CodeLabel()

    // handle the cases where the query is less than all keys (including the
    // empty index case), to establish the precondition of runQuery, and as a
    // fast path for a common case
    cb.if_(nKeys.ceq(0), {
      cb.assign(startIdx, 0L)
      cb.assign(endIdx, 0L)
      cb.goto(LReturn)
    })

    val rootChildren = readInternalNode(cb, rootOffset).loadField(cb, "children").get(cb).asIndexable
    val firstChild = rootChildren.loadElement(cb, 0).get(cb).asBaseStruct
    val firstKey = firstChild.loadField(cb, "first_key").get(cb).asBaseStruct
    val compStartWithFirstKey =
      compareStructWithPartitionIntervalEndpoint(cb, firstKey, startKey, startLeansRight)
    cb.if_(compStartWithFirstKey > 0, {
      cb.assign(startIdx, firstChild.loadField(cb, "first_idx").get(cb).asLong.value)
      cb.assign(startLeaf, getFirstLeaf(cb, firstChild))

      val compEndWithFirstKey =
        compareStructWithPartitionIntervalEndpoint(cb, firstKey, endKey, endLeansRight)
      cb.if_(compEndWithFirstKey > 0, {
        cb.assign(endIdx, startIdx)
      }, {
        queryBound(cb, endKey, endLeansRight, rootLevel, rootOffset, nKeys, startLeaf)
        cb.assign(endIdx, queryResultStartIndex)
      })
      cb.goto(LReturn)
    })

    val stackInterval = SStackInterval.construct(EmitValue.present(startKey), EmitValue.present(endKey), cb.memoize(!startLeansRight), endLeansRight)
    val (_startIdx, _startLeaf, _endIdx) = runQuery(cb,
      stackInterval, rootLevel, rootOffset, nKeys, startLeaf, isPointQuery = false)
    cb.assign(startIdx, _startIdx)
    cb.assign(startLeaf, _startLeaf)
    cb.assign(endIdx, _endIdx)

    cb.define(LReturn)

    val n = cb.memoize(endIdx - startIdx)
    cb.if_(n < 0L, cb._fatal("n less than 0: ", n.toS, ", startIdx=", startIdx.toS, ", endIdx=", endIdx.toS, ", query=", cb.strValue(interval)))
    cb.if_(n > 0L && startIdx >= nKeys, cb._fatal("bad start idx: ", startIdx.toS, ", nKeys=", nKeys.toS))

    SStackStruct.constructFromArgs(cb, null, TTuple(TInt64, TInt64, leafChildType.virtualType),
      EmitCode.present(cb.emb, primitive(startIdx)),
      EmitCode.present(cb.emb, primitive(endIdx)),
      EmitCode.fromI(cb.emb)(cb => IEmitCode(cb, n ceq 0L, startLeaf)))
  }

  private[this] def queryBound(cb: EmitCodeBuilder,
    endpoint: SBaseStructValue,
    leansRight: Value[Boolean],
    rootLevel: Value[Int],
    rootOffset: Value[Long],
    rootSuccessorIndex: Value[Long],
    rootSuccessorLeaf: SValue
  ): Unit = {
    cb.invokeVoid(
      cb.emb.ecb.getOrDefineEmitMethod("queryBound",
        FastSeq(endpoint.st.paramType, typeInfo[Boolean], typeInfo[Int], typeInfo[Long], typeInfo[Long], leafChildLocalType.paramType),
        UnitInfo
      ) { emb =>
        emb.emitWithBuilder { cb =>
          val endpoint = emb.getSCodeParam(1).asBaseStruct
          val leansRight = emb.getCodeParam[Boolean](2)
          val rootLevel = emb.getCodeParam[Int](3)
          val rootOffset = emb.getCodeParam[Long](4)
          val rootSuccessorIndex = emb.getCodeParam[Long](5)
          val rootSuccessorLeaf = emb.getSCodeParam(6)
          val interval = SStackInterval.construct(EmitValue.present(endpoint), EmitValue.present(endpoint), cb.memoize(!leansRight), leansRight)
          val (startIndex, startLeaf, _) = runQuery(cb, interval, rootLevel, rootOffset, rootSuccessorIndex, rootSuccessorLeaf, isPointQuery = true)
          cb.assign(queryResultStartIndex, startIndex)
          cb.assign(queryResultStartLeaf, startLeaf)
          Code._empty
        }
      }, cb.this_, endpoint, leansRight, rootLevel, rootOffset, rootSuccessorIndex, rootSuccessorLeaf)
  }

  // Supports both point and interval queries. If `isPointQuery`, end key
  // is ignored.
  // `rootSuccessorIndex` and `rootSuccessorLeaf` must contain the data of the
  // first record following the subtree rooted at `rootOffset`.
  // If this is the root of the index, so there is no following record,
  // `rootSuccessorIndex` must be `nKeys`, and `rootSuccessorLeaf` can be anything,
  // as it will never be accessed.
  private[this] def runQuery(cb: EmitCodeBuilder,
    interval: SStackIntervalValue,
    rootLevel: Value[Int],
    rootOffset: Value[Long],
    rootSuccessorIndex: Value[Long],
    rootSuccessorLeaf: SValue,
    isPointQuery: Boolean
  ): (Value[Long], SStackStructValue, Value[Long]) = {
    val startKey = interval.loadStart(cb).get(cb).asBaseStruct
    val startLeansRight = cb.memoize(!interval.includesStart)
    val endKey = interval.loadEnd(cb).get(cb).asBaseStruct
    val endLeansRight = interval.includesEnd

    def searchChildren(children: SIndexableValue, isInternalNode: Boolean): (Value[Int], Value[Int]) = {
      val keyFieldName = if (isInternalNode) "first_key" else "key"
      if (isPointQuery) {
        def ltNeedle(child: IEmitCode): Code[Boolean] = {
          val key = child.get(cb).asBaseStruct.loadField(cb, keyFieldName).get(cb).asBaseStruct
          val c = compareStructWithPartitionIntervalEndpoint(cb, key, startKey, startLeansRight)
          c < 0
        }
        val idx = BinarySearch.lowerBound(cb, children, ltNeedle)
        (idx, idx)
      } else {
        arrayOfStructFindIntervalRange(cb, children, startKey, startLeansRight, endKey, endLeansRight,
          _.get(cb).asBaseStruct.loadField(cb, keyFieldName))
      }
    }

    val startIndex: Settable[Long] = cb.newLocal[Long]("startIndex")
    val startLeaf: SStackStructSettable = cb.newSLocal(leafChildLocalType, "startOffset").asInstanceOf[SStackStructSettable]
    val endIndex: Settable[Long] = cb.newLocal[Long]("endIndex")

    val successorIndex: Settable[Long] = cb.newLocal[Long]("queryInterval_successorIndex", rootSuccessorIndex)
    val successorLeaf: SStackStructSettable = cb.newSLocal(leafChildLocalType, "successorLeaf").asInstanceOf[SStackStructSettable]
    cb.assign(successorLeaf, rootSuccessorLeaf)

    val level = cb.newLocal[Int]("queryInterval_level", rootLevel)
    val nodeOffset = cb.newLocal[Long]("queryInterval_nodeOffset", rootOffset)

    // loop invariants:
    // * `successorIndex` and `successorOffset` always point to the first record
    // to the right of the current subtree.
    // * `startKey` is always greater than the first key of the current subtree
    val Lstart = CodeLabel()
    cb.define(Lstart)

    def updateSuccessor(children: SIndexableValue, idx: Value[Int]): Unit = {
      cb.if_(idx < children.loadLength(), {
        val successorChild = children.loadElement(cb, idx).get(cb).asBaseStruct
        cb.assign(successorIndex, successorChild.loadField(cb, "first_idx").get(cb).asLong.value)
        cb.assign(successorLeaf, getFirstLeaf(cb, successorChild))
      })
    }

    cb.if_(level > 0, {
      /*
      InternalNode(
        children: IndexedSeq[InternalChild])
      InternalChild(
        index_file_offset: Long,
        first_idx: Long,
        first_key: Annotation,
        first_record_offset: Long,
        first_annotation: Annotation)
       */
      val children = readInternalNode(cb, nodeOffset).loadField(cb, "children").get(cb).asIndexable

      val (start, end) = searchChildren(children, isInternalNode = true)

      cb.assign(level, level-1)
      cb.if_(start.ceq(0) || end.ceq(0), cb._fatal("queryInterval broken invariant"))

      cb.if_(if (isPointQuery) const(true).get else start.ceq(end), {
        updateSuccessor(children, start)
        cb.assign(nodeOffset, children.loadElement(cb, start-1).get(cb).asBaseStruct.loadField(cb, "index_file_offset").get(cb).asLong.value)
        cb.goto(Lstart)
      })

      cb.if_(!(start < children.loadLength()), cb._fatal("unreachable"))

      // continue with separate point queries for each endpoint
      updateSuccessor(children, end)
      cb.assign(nodeOffset, children.loadElement(cb, end-1).get(cb).asBaseStruct.loadField(cb, "index_file_offset").get(cb).asLong.value)
      queryBound(cb, endKey, endLeansRight, level, nodeOffset, successorIndex, successorLeaf)
      cb.assign(endIndex, queryResultStartIndex)

      updateSuccessor(children, start)
      cb.assign(nodeOffset, children.loadElement(cb, start-1).get(cb).asBaseStruct.loadField(cb, "index_file_offset").get(cb).asLong.value)
      queryBound(cb, startKey, startLeansRight, level, nodeOffset, successorIndex, successorLeaf)
      cb.assign(startIndex, queryResultStartIndex)
      cb.assign(startLeaf, queryResultStartLeaf)
    }, {
      /*
      LeafNode(
        first_idx: Long,
        keys: IndexedSeq[LeafChild])
      LeafChild(
        key: Annotation,
        offset: Long,
        annotation: Annotation)
       */
      val node = readLeafNode(cb, nodeOffset).asBaseStruct
      val children = node.asBaseStruct.loadField(cb, "keys").get(cb).asIndexable

      val (start, end) = searchChildren(children, isInternalNode = false)

      val firstIndex = cb.memoize(node.asBaseStruct.loadField(cb, "first_idx")).get(cb).asInt64.value
      cb.if_(start < children.loadLength(), {
        cb.assign(startIndex, firstIndex + start.toL)
        cb.assign(startLeaf, children.loadElement(cb, start).get(cb).asBaseStruct.toStackStruct(cb))
      }, {
        cb.if_(successorIndex.cne(firstIndex + start.toL), cb._fatal("queryInterval broken invariant"))
        cb.assign(startIndex, successorIndex)
        cb.assign(startLeaf, successorLeaf)
      })
      cb.assign(endIndex, firstIndex + end.toL)
    })

    (startIndex, startLeaf, endIndex)
  }

  private[this] def getFirstLeaf(cb: EmitCodeBuilder, internalChild: SBaseStructValue): SValue = {
    new SStackStructValue(leafChildLocalType, Array(
      EmitValue.present(internalChild.loadField(cb, "first_key").get(cb)),
      EmitValue.present(internalChild.loadField(cb, "first_record_offset").get(cb)),
      EmitValue.present(internalChild.loadField(cb, "first_annotation").get(cb))))
  }

  // internal node is an array of children
  private[io] def readInternalNode(cb: EmitCodeBuilder, offset: Value[Long]): SBaseStructValue = {
    val ret = cb.newSLocal(internalPType.sType, "internalNode")

    // returns an address if cached, or 0L if not found
    val cached = cb.memoize(cache.invoke[Long, Long]("get", offset))

    cb.if_(cached cne 0L, {
      cb.assign(ret, internalPType.loadCheapSCode(cb, cached))
    }, {
      cb.assign(ret,
        cb.invokeSCode(
          cb.emb.ecb.getOrDefineEmitMethod("readInternalNode", FastSeq(LongInfo), ret.st.paramType) { emb =>
            emb.emitSCode { cb =>
              val offset = emb.getCodeParam[Long](1)
              cb += is.invoke[Long, Unit]("seek", offset)
              val ib = cb.memoize(internalCodec.buildCodeInputBuffer(is))
              cb.if_(ib.readByte() cne 1, cb._fatal("bad buffer at internal!"))
              val region = cb.memoize(cb.emb.ecb.pool().invoke[Region.Size, Region]("getRegion", Region.TINIER))
              val internalNode = internalDec.apply(cb, region, ib)
              val internalNodeAddr = internalPType.store(cb, region, internalNode, false)
              cb += cache.invoke[Long, Region, Long, Unit]("put", offset, region, internalNodeAddr)
              internalNode
            }
          },
          cb.this_,
          offset
        ))
    })

    ret.asBaseStruct
  }

  // leaf node is a struct
  private[io] def readLeafNode(cb: EmitCodeBuilder, offset: Value[Long]): SBaseStructValue = {
    val ret = cb.newSLocal(leafPType.sType, "leafNode")

    // returns an address if cached, or 0L if not found
    val cached = cb.memoize(cache.invoke[Long, Long]("get", offset))

    cb.if_(cached cne 0L, {
      cb.assign(ret, leafPType.loadCheapSCode(cb, cached))
    }, {
      cb.assign(ret, cb.invokeSCode(
        cb.emb.ecb.getOrDefineEmitMethod("readLeafNode", FastSeq(LongInfo), ret.st.paramType) { emb =>
          emb.emitSCode { cb =>
            val offset = emb.getCodeParam[Long](1)
            cb += is.invoke[Long, Unit]("seek", offset)
            val ib = cb.memoize(leafCodec.buildCodeInputBuffer(is))
            cb.if_(ib.readByte() cne 0, cb._fatal("bad buffer at leaf!"))
            val region = cb.memoize(cb.emb.ecb.pool().invoke[Region.Size, Region]("getRegion", Region.TINIER))
            val leafNode = leafDec.apply(cb, region, ib)
            val leafNodeAddr = leafPType.store(cb, region, leafNode, false)
            cb += cache.invoke[Long, Region, Long, Unit]("put", offset, region, leafNodeAddr)
            leafPType.loadCheapSCode(cb, leafNodeAddr)
          }
        },
        cb.this_,
        offset
      ))
    })
    ret.asBaseStruct
  }

  def queryIndex(cb: EmitCodeBuilder, region: Value[Region], absIndex: Value[Long]): SBaseStructValue = {
    cb.invokeSCode(
      cb.emb.ecb.getOrDefineEmitMethod("queryIndex",
        FastSeq(classInfo[Region], typeInfo[Long]),
        leafChildType.paramType
      ) { emb =>
        emb.emitSCode { cb =>

          val region = emb.getCodeParam[Region](1)
          val absIndex = emb.getCodeParam[Long](2)
          val level = cb.newLocal[Int]("lowerBound_level", metadata.invoke[Int]("height") - 1)
          val offset = cb.newLocal[Long]("lowerBound_offset", metadata.invoke[Long]("rootOffset"))
          val branchingFactor = cb.memoize(metadata.invoke[Int]("branchingFactor"))
          val result = cb.emb.newPLocal(leafChildType)

          cb.if_(absIndex >= nKeys(cb), cb._fatal("bad idx: ", absIndex.toS))

          val Lstart = CodeLabel()
          cb.define(Lstart)
          cb.if_(level ceq 0, {
            val leafNode = readLeafNode(cb, offset)
            val localIdx = cb.memoize((absIndex - leafNode.loadField(cb, "first_idx").get(cb).asInt64.value.toL).toI)
            cb.assign(result, leafNode.loadField(cb, "keys").get(cb).asIndexable.loadElement(cb, localIdx).get(cb))
          }, {
            val internalNode = readInternalNode(cb, offset)
            val children = internalNode.loadField(cb, "children").get(cb).asIndexable
            val firstIdx = children.loadElement(cb, 0).get(cb).asBaseStruct.loadField(cb, "first_idx").get(cb).asInt64.value
            val nKeysPerChild = cb.memoize(Code.invokeStatic2[java.lang.Math, Double, Double, Double]("pow",
              branchingFactor.toD,
              level.toD).toL)
            val localIdx = cb.memoize((absIndex - firstIdx) / nKeysPerChild)
            cb.assign(level, level - 1)
            cb.assign(offset, children.loadElement(cb, localIdx.toI).get(cb).asBaseStruct.loadField(cb, "index_file_offset").get(cb).asInt64.value)
            cb.goto(Lstart)
          })

          leafChildType.coerceOrCopy(cb, region, result, false)
        }
      },
      cb.this_,
      region,
      absIndex
    ).asBaseStruct
  }
}
