package is.hail.io.index

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.functions.IntervalFunctions.compareStructWithPartitionIntervalEndpoint
import is.hail.expr.ir.{BinarySearch, EmitCode, EmitCodeBuilder, EmitMethodBuilder, EmitSettable}
import is.hail.io.fs.FS
import is.hail.rvd.AbstractIndexSpec
import is.hail.types.physical.stypes.concrete.SStackStruct
import is.hail.types.physical.stypes.interfaces.{SBaseStructValue, SIntervalValue, primitive}
import is.hail.types.physical.stypes.primitives.{SBooleanValue, SInt64}
import is.hail.types.physical.stypes.{EmitType, SSettable}
import is.hail.types.physical.{PCanonicalArray, PCanonicalBaseStruct}
import is.hail.types.virtual.{TBoolean, TInt64, TTuple}
import is.hail.utils._

import java.io.InputStream

case class VariableMetadata(
  branchingFactor: Int,
  height: Int,
  nKeys: Long,
  rootOffset: Long,
  attributes: Map[String, Any]
)


class StagedIndexReader(emb: EmitMethodBuilder[_], spec: AbstractIndexSpec) {
  private[this] val cache: Settable[LongToRegionValueCache] = emb.genFieldThisRef[LongToRegionValueCache]("index_cache")
  private[this] val metadata: Settable[VariableMetadata] = emb.genFieldThisRef[VariableMetadata]("index_file_metadata")

  private[this] val is: Settable[ByteTrackingInputStream] = emb.genFieldThisRef[ByteTrackingInputStream]("index_is")

  private[this] val leafPType = spec.leafCodec.encodedType.decodedPType(spec.leafCodec.encodedVirtualType)
  private[this] val internalPType = spec.internalNodeCodec.encodedType.decodedPType(spec.internalNodeCodec.encodedVirtualType)
  private[this] val leafDec = spec.leafCodec.encodedType.buildDecoder(spec.leafCodec.encodedVirtualType, emb.ecb)
  private[this]val internalDec = spec.internalNodeCodec.encodedType.buildDecoder(spec.internalNodeCodec.encodedVirtualType, emb.ecb)

  private[this] val leafChildType = leafPType.asInstanceOf[PCanonicalBaseStruct].types(1).asInstanceOf[PCanonicalArray].elementType.sType
  val leafChildEmitType = EmitType(leafChildType, false)
  private[this] val queryType = SStackStruct(TTuple(TInt64, leafChildType.virtualType), FastIndexedSeq(EmitType(SInt64, true), leafChildEmitType))

  def initialize(cb: EmitCodeBuilder,
    indexPath: Value[String]
  ): Unit = {
    val fs = cb.emb.getFS
    cb.assign(cache, Code.newInstance[LongToRegionValueCache, Int](16))
    cb.assign(metadata, Code.invokeScalaObject2[FS, String, IndexMetadataUntypedJSON](
      IndexReader.getClass, "readUntyped", fs, indexPath
    ).invoke[VariableMetadata]("toFileMetadata"))

    // FIXME: hardcoded. Will break if we change spec -- assumption not introduced with this code, but propagated.
    cb.assign(is, Code.newInstance[ByteTrackingInputStream, InputStream](cb.emb.open(indexPath.concat("/index"), false)))

  }

  def close(cb: EmitCodeBuilder): Unit = {
    cb += is.invoke[Unit]("close")
    cb += cache.invoke[Unit]("free")
    cb.assign(is, Code._null)
    cb.assign(cache, Code._null)
    cb.assign(metadata, Code._null)
  }

  /**
   * returns tuple of (start index, end index, starting leaf)
   * memory of starting leaf is not owned by `region`, consumers should deep copy if necessary
   * starting leaf may be missing if the index is empty
   */
  def queryInterval(cb: EmitCodeBuilder,
    region: Value[Region],
    interval: SIntervalValue): SBaseStructValue = {

    val start = interval.loadStart(cb).get(cb).asBaseStruct
    val end = interval.loadEnd(cb).get(cb).asBaseStruct
    val includesStart = interval.includesStart()
    val includesEnd = interval.includesEnd()

    val startQuerySettable = cb.newSLocal(queryType, "startQuery")
    cb.ifx(includesStart,
      cb.assign(startQuerySettable, queryBound(cb, region, start, primitive(false), "lower")),
      cb.assign(startQuerySettable, queryBound(cb, region, start, primitive(true), "upper"))
    )

    val endQuerySettable = cb.newSLocal(queryType, "endQuery")
    cb.ifx(includesEnd,
      cb.assign(endQuerySettable, queryBound(cb, region, end, primitive(true), "upper")),
      cb.assign(endQuerySettable, queryBound(cb, region, end, primitive(false), "lower"))
    )

    val startIdx = cb.memoize(startQuerySettable.asBaseStruct.loadField(cb, 0).get(cb).asInt64.value)
    val endIdx = cb.memoize(endQuerySettable.asBaseStruct.loadField(cb, 0).get(cb).asInt64.value)
    val n = cb.memoize(endIdx - startIdx)
    val startLeaf = cb.memoize(startQuerySettable.asBaseStruct.loadField(cb, 1))

    cb.ifx(n < 0L, cb._fatal("n less than 0: ", n.toS, ", startQuery=", cb.strValue(startQuerySettable), ", endQuery=", cb.strValue(endQuerySettable)))


    SStackStruct.constructFromArgs(cb, region, TTuple(TInt64, TInt64, startLeaf.st.virtualType), EmitCode.present(cb.emb, primitive(startIdx)), EmitCode.present(cb.emb, primitive(endIdx)), startLeaf)
  }

  // internal node is an array of children
  private[io] def readInternalNode(cb: EmitCodeBuilder, offset: Value[Long]): SBaseStructValue = {
    val ret = cb.newSLocal(internalPType.sType, "internalNode")

    // returns an address if cached, or -1L if not found
    val cached = cb.memoize(cache.invoke[Long, Long]("get", offset))

    cb.ifx(cached cne -1L, {
      cb.assign(ret, internalPType.loadCheapSCode(cb, cached))
    }, {
      cb.assign(ret, cb.invokeSCode(cb.emb.ecb.getOrGenEmitMethod("readInternalNode", ("readInternalNode", this), FastIndexedSeq(LongInfo), ret.st.paramType) { emb =>
        emb.emitSCode { cb =>
          val offset = emb.getCodeParam[Long](1)
          cb += is.invoke[Long, Unit]("seek", offset)
          val ib = cb.memoize(spec.internalNodeCodec.buildCodeInputBuffer(is))
          cb.ifx(ib.readByte() cne 1, cb._fatal("bad buffer at internal!"))
          val region = cb.memoize(cb.emb.ecb.pool().invoke[Region.Size, Region]("getRegion", Region.TINIER))
          val internalNode = internalDec.apply(cb, region, ib)
          val internalNodeAddr = internalPType.store(cb, region, internalNode, false)
          cb += cache.invoke[Long, Region, Long, Unit]("put", offset, region, internalNodeAddr)
          internalNode
        }
      }, offset))
    })

    ret.asBaseStruct
  }

  // leaf node is a struct
  private[io] def readLeafNode(cb: EmitCodeBuilder, offset: Value[Long]): SBaseStructValue = {
    val ret = cb.newSLocal(leafPType.sType, "leafNode")

    // returns an address if cached, or -1L if not found
    val cached = cb.memoize(cache.invoke[Long, Long]("get", offset))

    cb.ifx(cached cne -1L, {
      cb.assign(ret, leafPType.loadCheapSCode(cb, cached))
    }, {
      cb.assign(ret, cb.invokeSCode(cb.emb.ecb.getOrGenEmitMethod("readLeafNode", ("readLeafNode", this), FastIndexedSeq(LongInfo), ret.st.paramType) { emb =>
        emb.emitSCode { cb =>
          val offset = emb.getCodeParam[Long](1)
          cb += is.invoke[Long, Unit]("seek", offset)
          val ib = cb.memoize(spec.leafCodec.buildCodeInputBuffer(is))
          cb.ifx(ib.readByte() cne 0, cb._fatal("bad buffer at leaf!"))
          val region = cb.memoize(cb.emb.ecb.pool().invoke[Region.Size, Region]("getRegion", Region.TINIER))
          val leafNode = leafDec.apply(cb, region, ib)
          val leafNodeAddr = leafPType.store(cb, region, leafNode, false)
          cb += cache.invoke[Long, Region, Long, Unit]("put", offset, region, leafNodeAddr)
          leafPType.loadCheapSCode(cb, leafNodeAddr)
        }
      }, offset))
    })
    ret.asBaseStruct
  }

  // returns queryType
  def queryBound(cb: EmitCodeBuilder, region: Value[Region], partitionBoundLeftEndpoint: SBaseStructValue, leansRight: SBooleanValue, boundType: String): SBaseStructValue = {
    cb.invokeSCode(
      cb.emb.ecb.getOrGenEmitMethod("lowerBound",
        ("lowerBound", this, boundType),
        FastIndexedSeq(typeInfo[Region], partitionBoundLeftEndpoint.st.paramType, leansRight.st.paramType),
      queryType.paramType) { emb =>
      emb.emitSCode { cb =>
        val region = emb.getCodeParam[Region](1)
        val endpoint = emb.getSCodeParam(2).asBaseStruct
        val leansRight = emb.getSCodeParam(3).asBoolean
        queryType.coerceOrCopy(cb, region, queryBound(cb, region, endpoint, leansRight, cb.memoize(metadata.invoke[Int]("height") - 1), cb.memoize(metadata.invoke[Long]("rootOffset")), boundType), false) }
    }, region, partitionBoundLeftEndpoint, leansRight).asBaseStruct
  }

  // partitionBoundEndpoint is a tuple(partitionBoundEndpoint, bool)
  // returns a tuple of (index, LeafChild)
  private def queryBound(cb: EmitCodeBuilder,
    region: Value[Region],
    endpoint: SBaseStructValue,
    leansRight: SBooleanValue,
    level: Value[Int],
    offset: Value[Long],
    boundType: String): SBaseStructValue = {

    val rInd: Settable[Long] = cb.newLocal[Long]("lowerBoundIndex")
    val rLeafChild: EmitSettable = cb.emb.newEmitLocal(leafChildEmitType)

    val levelSettable = cb.newLocal[Int]("lowerBound_level")
    val offsetSettable = cb.newLocal[Long]("lowerBound_offset")

    cb.assign(levelSettable,level)
    cb.assign(offsetSettable,offset)

    val boundAndSignTuple = SStackStruct.constructFromArgs(cb,
      region,
      TTuple(endpoint.st.virtualType, TBoolean),
      EmitCode.present(cb.emb, endpoint),
      EmitCode.present(cb.emb, leansRight)
    )

    val Lstart = CodeLabel()
    cb.define(Lstart)

    cb.ifx(levelSettable ceq 0, {
      val node = readLeafNode(cb, offsetSettable).asBaseStruct

      /*
      LeafNode(
        firstIndex: Long,
        children: IndexedSeq[LeafChild]
      LeafChild(
        key: Annotation,
        recordOffset: Long,
        annotation: Annotation)
       */
      val children = node.asBaseStruct.loadField(cb, "keys").get(cb).asIndexable

      val idx = new BinarySearch(cb.emb,
        children.st,
        EmitType(boundAndSignTuple.st, true),
        ((cb, elt) => cb.memoize(elt.get(cb).asBaseStruct.loadField(cb, "key"))),
        bound=boundType,
        ltF = { (cb, containerEltEV, partBoundEV) =>
          val containerElt = containerEltEV.get(cb).asBaseStruct
          val partBound = partBoundEV.get(cb).asBaseStruct
          val endpoint = partBound.loadField(cb, 0).get(cb).asBaseStruct
          val leansRight = partBound.loadField(cb, 1).get(cb).asBoolean.value
          val comp = compareStructWithPartitionIntervalEndpoint(cb, containerElt, endpoint, leansRight)
          val ltOrGt = cb.memoize(if (boundType == "lower") comp < 0 else comp > 0)
          ltOrGt
        }
      )
        .search(cb, children, EmitCode.present(cb.emb, boundAndSignTuple))

      val firstIndex = node.asBaseStruct.loadField(cb, "first_idx").get(cb).asInt64.value.get
      val updatedIndex = firstIndex + idx.toL
      cb.assign(rInd, updatedIndex)
      val idxWithModification = cb.memoize((if (boundType == "lower") idx else cb.memoize(idx-1)) min children.loadLength())
      val leafChild = children.loadElement(cb, idxWithModification).get(cb).asBaseStruct
      cb.assign(rLeafChild, EmitCode.present(cb.emb, leafChild))
    }, {
      val children = readInternalNode(cb, offsetSettable).loadField(cb, "children").get(cb).asIndexable
      cb.ifx(children.loadLength() ceq 0, {
        // empty interal node occurs if the indexed file contains no keys
        cb.assign(rInd, 0L)
        cb.assign(rLeafChild, EmitCode.missing(cb.emb, leafChildType))
      }, {
        val idx = new BinarySearch(cb.emb,
          children.st,
          EmitType(boundAndSignTuple.st, true),
          ((cb, elt) => cb.memoize(elt.get(cb).asBaseStruct.loadField(cb, "first_key"))),
          bound=boundType,
          ltF = { (cb, containerEltEV, partBoundEV) =>
            val containerElt = containerEltEV.get(cb).asBaseStruct
            val partBound = partBoundEV.get(cb).asBaseStruct
            val endpoint = partBound.loadField(cb, 0).get(cb).asBaseStruct
            val leansRight = partBound.loadField(cb, 1).get(cb).asBoolean.value
            val comp = compareStructWithPartitionIntervalEndpoint(cb, containerElt, endpoint, leansRight)
            val ltOrGt = cb.memoize(if (boundType == "lower") comp < 0 else comp > 0)
            ltOrGt
          }
        )
          .search(cb, children, EmitCode.present(cb.emb, boundAndSignTuple))
        cb.assign(levelSettable, levelSettable-1)
        cb.assign(offsetSettable, children.loadElement(cb, (idx-1).max(0)).get(cb).asBaseStruct.loadField(cb, "index_file_offset").get(cb).asLong.value)
        cb.goto(Lstart)
      })
    })

    SStackStruct.constructFromArgs(cb, region, queryType.virtualType,
      EmitCode.present(cb.emb, primitive(rInd)),
      rLeafChild)
  }
}
