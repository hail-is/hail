package is.hail.io.index

import java.io.OutputStream

import is.hail.annotations.{Annotation, Region, RegionValueBuilder, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{CodeParam, EmitClassBuilder, EmitCodeBuilder, EmitFunctionBuilder, EmitMethodBuilder, ExecuteContext, IEmitCode, ParamType, coerce}
import is.hail.expr.types
import is.hail.expr.types._
import is.hail.expr.types.encoded.EType
import is.hail.expr.types.physical.{PBaseStruct, PBaseStructValue, PCanonicalArray, PCanonicalBaseStructSettable, PCanonicalStruct, PCode, PInt64, PType}
import is.hail.expr.types.virtual.Type
import is.hail.io.fs.FS
import is.hail.io._
import is.hail.rvd.AbstractRVDSpec
import is.hail.utils._
import is.hail.utils.richUtils.ByteTrackingOutputStream
import org.json4s.Formats
import org.json4s.jackson.Serialization

trait AbstractIndexMetadata {
  def fileVersion: Int

  def branchingFactor: Int

  def height: Int

  def keyType: Type

  def annotationType: Type

  def nKeys: Long

  def indexPath: String

  def rootOffset: Long

  def attributes: Map[String, Any]
}

case class IndexMetadataUntypedJSON(
  fileVersion: Int,
  branchingFactor: Int,
  height: Int,
  nKeys: Long,
  indexPath: String,
  rootOffset: Long,
  attributes: Map[String, Any]
) {
  def toMetadata(keyType: Type, annotationType: Type): IndexMetadata = IndexMetadata(
    fileVersion, branchingFactor,
    height, keyType, annotationType,
    nKeys, indexPath, rootOffset, attributes)
}

case class IndexMetadata(
  fileVersion: Int,
  branchingFactor: Int,
  height: Int,
  keyType: Type,
  annotationType: Type,
  nKeys: Long,
  indexPath: String,
  rootOffset: Long,
  attributes: Map[String, Any]
) extends AbstractIndexMetadata

case class IndexNodeInfo(
  indexFileOffset: Long,
  firstIndex: Long,
  firstKey: Annotation,
  firstRecordOffset: Long,
  firstAnnotation: Annotation
)

object IndexWriter {
  val version: SemanticVersion = SemanticVersion(1, 1, 0)

  val spec: BufferSpec = BufferSpec.default
  def builder(
    ctx: ExecuteContext,
    keyType: PType,
    annotationType: PType,
    branchingFactor: Int = 4096,
    attributes: Map[String, Any] = Map.empty[String, Any]
  ): String => IndexWriter = {
    val f = StagedIndexPartitionWriter.build(ctx, keyType, annotationType, branchingFactor, attributes);
    { path: String =>
      new IndexWriter(keyType, annotationType, f(path))
    }
  }
}

class IndexWriter(keyType: PType, valueType: PType, comp: CompiledIndexWriter) extends AutoCloseable {
  private val region = Region()
  private val rvb = new RegionValueBuilder(region)
  def +=(x: Annotation, offset: Long, annotation: Annotation): Unit = {
    rvb.start(keyType)
    rvb.addAnnotation(keyType.virtualType, x)
    val koff = rvb.end()
    rvb.start(valueType)
    rvb.addAnnotation(valueType.virtualType, annotation)
    val voff = rvb.end()
    comp.apply(koff, offset, voff)
  }

  def close(): Unit = {
    region.close()
    comp.close()
  }
}

class IndexWriterArrayBuilder(name: String, maxSize: Int, sb: SettableBuilder, region: Value[Region], arrayType: PCanonicalArray) {
  private val aoff = sb.newSettable[Long](s"${name}_aoff")
  private val len = sb.newSettable[Int](s"${name}_len")

  val eltType: PCanonicalStruct = types.coerce[PCanonicalStruct](arrayType.elementType)
  private val elt = new PCanonicalBaseStructSettable(eltType, sb.newSettable[Long](s"${name}_elt_off"))

  def length: Code[Int] = len

  def loadFrom(cb: EmitCodeBuilder, a: Code[Long], l: Code[Int]): Unit = {
    cb += (aoff := a)
    cb += (len := l)
  }

  def create(cb: EmitCodeBuilder, dest: Code[Long]): Unit = {
    cb += (aoff := arrayType.allocate(region, maxSize))
    cb += arrayType.stagedInitialize(aoff, maxSize)
    cb += PCode(arrayType, aoff).store(cb.emb, region, dest)
    cb += (len := 0)
  }

  def storeLength(cb: EmitCodeBuilder): Unit = cb += arrayType.storeLength(aoff, length)

  def setFieldValue(cb: EmitCodeBuilder, name: String, field: PCode): Unit = {
    cb += eltType.setFieldPresent(elt.a, name)
    cb += StagedRegionValueBuilder.deepCopy(cb.emb.ecb, region, eltType.fieldType(name), field.code, eltType.fieldOffset(elt.a, name))
  }

  def setField(cb: EmitCodeBuilder, name: String, v: => IEmitCode): Unit =
    v.consume(cb,
      cb += eltType.setFieldMissing(elt.a, name),
      setFieldValue(cb, name, _))

  def addChild(cb: EmitCodeBuilder): Unit = {
    loadChild(cb, len)
    cb += (len := len + 1)
  }
  def loadChild(cb: EmitCodeBuilder, idx: Code[Int]): Unit = cb += elt.store(PCode(eltType, arrayType.elementOffset(aoff, idx)))
  def getLoadedChild: PBaseStructValue = elt

  def getChild(idx: Value[Int]): PCode = PCode(eltType, arrayType.elementOffset(aoff, idx))
}

class StagedLeafNodeBuilder(maxSize: Int, keyType: PType, annotationType: PType, sb: SettableBuilder) {
  private val region = sb.newSettable[Region]("leaf_node_region")
  val ab = new IndexWriterArrayBuilder("leaf_node", maxSize,
    sb, region,
    LeafNodeBuilder.arrayType(keyType, annotationType))

  val pType: PCanonicalStruct = LeafNodeBuilder.typ(keyType, annotationType)
  private val node = new PCanonicalBaseStructSettable(pType, sb.newSettable[Long]("lef_node_addr"))

  def close(cb: EmitCodeBuilder): Unit = cb.ifx(!region.isNull, cb += region.invalidate())

  def reset(cb: EmitCodeBuilder, firstIdx: Code[Long]): Unit = {
    cb += region.invoke[Unit]("clear")
    cb += node.store(PCode(pType, pType.allocate(region)))
    cb += PInt64().storePrimitiveAtAddress(pType.fieldOffset(node.a, "first_idx"), PInt64(), firstIdx)
    ab.create(cb, pType.fieldOffset(node.a, "keys"))
  }

  def create(cb: EmitCodeBuilder, firstIdx: Code[Long]): Unit = {
    cb += (region := Region.stagedCreate(Region.REGULAR))
    cb += node.store(PCode(pType, pType.allocate(region)))
    cb += PInt64().storePrimitiveAtAddress(pType.fieldOffset(node.a, "first_idx"), PInt64(), firstIdx)
    ab.create(cb, pType.fieldOffset(node.a, "keys"))
  }

  def encode(cb: EmitCodeBuilder, ob: Value[OutputBuffer]): Unit = {
    val enc = EType.defaultFromPType(pType).buildEncoder(pType, cb.emb.ecb)
    ab.storeLength(cb)
    cb += enc(node.a, ob)
  }

  def nodeAddress: PBaseStructValue = node

  def add(cb: EmitCodeBuilder, key: => IEmitCode, offset: Code[Long], annotation: => IEmitCode): Unit = {
    ab.addChild(cb)
    ab.setField(cb, "key", key)
    ab.setFieldValue(cb, "offset", PCode(PInt64(), offset))
    ab.setField(cb, "annotation", annotation)
  }

  def loadChild(cb: EmitCodeBuilder, idx: Code[Int]): Unit = ab.loadChild(cb, idx)
  def getLoadedChild: PBaseStructValue = ab.getLoadedChild
  def firstIdx: PCode = PInt64().load(pType.fieldOffset(node.a, "first_idx"))
}

class StagedInternalNodeBuilder(maxSize: Int, keyType: PType, annotationType: PType, sb: SettableBuilder) {
  private val region = sb.newSettable[Region]("internal_node_region")
  val ab = new IndexWriterArrayBuilder("internal_node", maxSize,
    sb, region,
    InternalNodeBuilder.arrayType(keyType, annotationType))

  val pType: PCanonicalStruct = InternalNodeBuilder.typ(keyType, annotationType)
  private val node = new PCanonicalBaseStructSettable(pType, sb.newSettable[Long]("internal_node_node"))

  def loadFrom(cb: EmitCodeBuilder, ib: StagedInternalNodeArrayBuilder, idx: Value[Int]): Unit = {
    cb += (region := ib.getRegion(idx))
    cb += (node.a := ib.getArrayOffset(idx))
    val aoff = node.loadField(cb, 0).handle(cb, ()).tcode[Long]
    ab.loadFrom(cb, aoff, ib.getLength(idx))
  }

  def store(cb: EmitCodeBuilder, ib: StagedInternalNodeArrayBuilder, idx: Value[Int]): Unit = {
    cb += ib.update(idx, region.get, node.a.get, ab.length)
  }

  def clear(cb: EmitCodeBuilder): Unit = { cb += region.invoke[Unit]("clear") }

  def allocate(cb: EmitCodeBuilder): Unit = {
    cb += node.store(PCode(pType, pType.allocate(region)))
    ab.create(cb, pType.fieldOffset(node.a, "children"))
  }

  def create(cb: EmitCodeBuilder): Unit = {
    cb += (region := Region.stagedCreate(Region.REGULAR))
    allocate(cb)
  }

  def encode(cb: EmitCodeBuilder, ob: Value[OutputBuffer]): Unit = {
    val enc = EType.defaultFromPType(pType).buildEncoder(pType, cb.emb.ecb)
    ab.storeLength(cb)
    cb += enc(node.a, ob)
  }

  def nodeAddress: PBaseStructValue = node

  def add(cb: EmitCodeBuilder, indexFileOffset: Code[Long], firstIndex: Code[Long], firstChild: PBaseStructValue): Unit = {
    val childtyp = types.coerce[PBaseStruct](firstChild.pt)
    ab.addChild(cb)
    ab.setFieldValue(cb, "index_file_offset", PCode(PInt64(), indexFileOffset))
    ab.setFieldValue(cb, "first_idx", PCode(PInt64(), firstIndex))
    ab.setField(cb, "first_key", firstChild.loadField(cb, childtyp.fieldIdx("key")))
    ab.setField(cb, "first_record_offset", firstChild.loadField(cb, childtyp.fieldIdx("offset")))
    ab.setField(cb, "first_annotation", firstChild.loadField(cb, childtyp.fieldIdx("annotation")))
  }

  def add(cb: EmitCodeBuilder, indexFileOffset: Code[Long], firstChild: PBaseStructValue): Unit = {
    val childtyp = types.coerce[PBaseStruct](firstChild.pt)
    ab.addChild(cb)
    ab.setFieldValue(cb, "index_file_offset", PCode(PInt64(), indexFileOffset))
    ab.setField(cb, "first_idx", firstChild.loadField(cb, childtyp.fieldIdx("first_idx")))
    ab.setField(cb, "first_key", firstChild.loadField(cb, childtyp.fieldIdx("first_key")))
    ab.setField(cb, "first_record_offset", firstChild.loadField(cb, childtyp.fieldIdx("first_record_offset")))
    ab.setField(cb, "first_annotation", firstChild.loadField(cb, childtyp.fieldIdx("first_annotation")))
  }

  def loadChild(cb: EmitCodeBuilder, idx: Code[Int]): Unit = ab.loadChild(cb, idx)
  def getLoadedChild: PBaseStructValue = ab.getLoadedChild
}

class StagedInternalNodeArrayBuilder(ib: Settable[InternalNodeArrayBuilder]) {
  def create(initSize: Code[Int]): Code[Unit] = ib := Code.newInstance[InternalNodeArrayBuilder, Int](initSize)
  def size: Code[Int] = ib.invoke[Int]("size")
  def add(r: Code[Region], aoff: Code[Long], len: Code[Int]): Code[Unit] =
    ib.invoke[Region, Long, Int, Unit]("add", r, aoff, len)

  def update(idx: Code[Int], r: Code[Region], aoff: Code[Long], len: Code[Int]): Code[Unit] =
    ib.invoke[Int, Region, Long, Int, Unit]("update", idx, r, aoff, len)

  def getRegion(idx: Code[Int]): Code[Region] = ib.invoke[Int, Region]("getRegion", idx)
  def getArrayOffset(idx: Code[Int]): Code[Long] = ib.invoke[Int, Long]("getArrayOffset", idx)
  def getLength(idx: Code[Int]): Code[Int] = ib.invoke[Int, Int]("getLength", idx)
  def close(): Code[Unit] = ib.invoke[Unit]("close")
}

class InternalNodeArrayBuilder(initSize: Int) {
  val rBuilder = new ArrayBuilder[Region](initSize)
  val aBuilder = new ArrayBuilder[Long](initSize)
  val lBuilder = new ArrayBuilder[Int](initSize)

  def size: Int = rBuilder.size

  def add(r: Region, aoff: Long, len: Int): Unit = {
    rBuilder += r
    aBuilder += aoff
    lBuilder += len
  }

  def update(idx: Int, r: Region, aoff: Long, len: Int): Unit = {
    if (idx == size) {
      add(r, aoff, len)
    } else {
      rBuilder.update(idx, r)
      aBuilder.update(idx, aoff)
      lBuilder.update(idx, len)
    }
  }

  def getRegion(idx: Int): Region = rBuilder(idx)
  def getArrayOffset(idx: Int): Long = aBuilder(idx)
  def getLength(idx: Int): Int = lBuilder(idx)

  def close(): Unit = rBuilder.result().foreach { r => r.close() }
}

case class StagedIndexMetadata(
  fileVersion: Int,
  branchingFactor: Int,
  keyType: Type,
  annotationType: Type,
  indexPath: String,
  attributes: Map[String, Any]
) {
  def serialize(out: OutputStream, height: Int, rootOffset: Long, nKeys: Long) {
    import AbstractRVDSpec.formats
    val metadata = IndexMetadata(fileVersion, branchingFactor, height, keyType, annotationType, nKeys, indexPath, rootOffset, attributes)
    Serialization.write(metadata, out)
  }
}

class IndexWriterUtils(path: String, fs: FS) {
  val indexPath: String = path + "/index"
  val metadataPath: String = path + "/metadata.json.gz"
  val trackedOS: ByteTrackingOutputStream = new ByteTrackingOutputStream(fs.create(indexPath))

  def bytesWritten: Long = trackedOS.bytesWritten
  def os: OutputStream = trackedOS

  def writeMetadata(meta: StagedIndexMetadata, height: Int, rootOffset: Long, nKeys: Long): Unit = {
    using(fs.create(metadataPath)) { os => meta.serialize(os, height, rootOffset, nKeys) }
  }

  def close(): Unit = {
    trackedOS.close()
  }
}

trait CompiledIndexWriter {
  def init(path: String): Unit
  def apply(x: Long, offset: Long, annotation: Long): Unit
  def close(): Unit
}

object StagedIndexPartitionWriter {
  def build(
    ctx: ExecuteContext,
    keyType: PType,
    annotationType: PType,
    branchingFactor: Int = 4096,
    attributes: Map[String, Any] = Map.empty[String, Any]
  ): String => CompiledIndexWriter = {
    val fb = EmitFunctionBuilder[CompiledIndexWriter](ctx, "indexwriter",
      FastIndexedSeq[ParamType](typeInfo[Long], typeInfo[Long], typeInfo[Long]),
      typeInfo[Unit])
    val cb = fb.ecb
    val siw = new StagedIndexPartitionWriter(branchingFactor, keyType, annotationType, attributes, cb)
    val init = cb.newEmitMethod("init", FastIndexedSeq[ParamType](typeInfo[String]), typeInfo[Unit])
    val close = cb.newEmitMethod("close", FastIndexedSeq[ParamType](), typeInfo[Unit])

    init.emitWithBuilder { cb =>
      val path = init.getCodeParam[String](1)
      siw.init(cb, path)
      Code._empty
    }

    fb.emitWithBuilder { cb =>
      val key = fb.getCodeParam[Long](1)
      val offset = fb.getCodeParam[Long](2)
      val annotation = fb.getCodeParam[Long](3)
      siw.add(cb, IEmitCode(cb, false, PCode(keyType, key)), offset, IEmitCode(cb, false, PCode(annotationType, annotation)))
      Code._empty
    }

    close.emitWithBuilder { cb =>
      siw.close(cb)
      Code._empty
    }
    val makeFB = fb.resultWithIndex()

    { path: String =>
      val f = makeFB(0, null)
      f.init(path)
      f
    }
  }
}

class StagedIndexPartitionWriter(branchingFactor: Int, keyType: PType, annotationType: PType, attributes: Map[String, Any], cb: EmitClassBuilder[_]) {
  require(branchingFactor > 1)

  private var elementIdx = cb.genFieldThisRef[Long]()
  private val utils = cb.genFieldThisRef[IndexWriterUtils]()
  private val metaOS = cb.genFieldThisRef[OutputStream]()
  private val ob = cb.genFieldThisRef[OutputBuffer]()
  private val internalAB = new StagedInternalNodeArrayBuilder(cb.genFieldThisRef[InternalNodeArrayBuilder]())

  private val metadata = cb.getObject(StagedIndexMetadata(
    IndexWriter.version.rep,
    branchingFactor,
    keyType.virtualType,
    annotationType.virtualType,
    "index",
    attributes))
  private val leafBuilder = new StagedLeafNodeBuilder(branchingFactor, keyType, annotationType, cb.fieldBuilder)

  private val writeInternalNode: EmitMethodBuilder[_] = {
    val m = cb.genEmitMethod[Int, Boolean, Unit]("writeInternalNode")

    val internalBuilder = new StagedInternalNodeBuilder(branchingFactor, keyType, annotationType, m.localBuilder)
    val parentBuilder = new StagedInternalNodeBuilder(branchingFactor, keyType, annotationType, m.localBuilder)

    m.emitWithBuilder { cb =>
      val level = m.getCodeParam[Int](1)
      val isRoot = m.getCodeParam[Boolean](2)
      val idxOff = cb.newLocal[Long]("indexOff")
      cb += (idxOff := utils.invoke[Long]("bytesWritten"))
      internalBuilder.loadFrom(cb, internalAB, level)
      cb += ob.writeByte(1.toByte)
      internalBuilder.encode(cb, ob)
      cb += ob.flush()

      val next = m.newLocal[Int]("next")
      cb += (next := level + 1)

      //if !isRoot
      cb.ifx(!isRoot, {
        cb.ifx(internalAB.size.ceq(next),
          parentBuilder.create(cb), {
            cb.ifx(internalAB.getLength(next).ceq(branchingFactor),
              cb += m.invokeCode[Unit](CodeParam(next), CodeParam(false)))
            parentBuilder.loadFrom(cb, internalAB, next)
          })
        internalBuilder.loadChild(cb, 0)
        parentBuilder.add(cb, idxOff, internalBuilder.getLoadedChild)
        parentBuilder.store(cb, internalAB, next)
      })

      internalBuilder.clear(cb)
      internalBuilder.allocate(cb)
      internalBuilder.store(cb, internalAB, level)
      Code._empty
    }
    m
  }

  private val writeLeafNode: EmitMethodBuilder[_] = {
    val m = cb.genEmitMethod[Unit]("writeLeafNode")

    val parentBuilder = new StagedInternalNodeBuilder(branchingFactor, keyType, annotationType, m.localBuilder)
    m.emitWithBuilder { cb =>
      val idxOff = cb.newLocal[Long]("indexOff")
      cb += (idxOff := utils.invoke[Long]("bytesWritten"))
      cb += ob.writeByte(0.toByte)
      leafBuilder.encode(cb, ob)
      cb += ob.flush()

      cb.ifx(internalAB.getLength(0).ceq(branchingFactor),
        cb += writeInternalNode.invokeCode[Unit](CodeParam(0), CodeParam(false)))
      parentBuilder.loadFrom(cb, internalAB, 0)

      leafBuilder.loadChild(cb, 0)
      parentBuilder.add(cb, idxOff, leafBuilder.firstIdx.tcode[Long], leafBuilder.getLoadedChild)
      parentBuilder.store(cb, internalAB, 0)
      leafBuilder.reset(cb, elementIdx)
      Code._empty
    }
    m
  }

  private val flush: EmitMethodBuilder[_] = {
    val m = cb.genEmitMethod[Long]("flush")
    m.emitWithBuilder { cb =>
      val idxOff = cb.newLocal[Long]("indexOff")
      val level = m.newLocal[Int]("level")
      cb.ifx(leafBuilder.ab.length > 0, cb += writeLeafNode.invokeCode[Unit]())
      cb += (level := 0)
      cb.whileLoop(level < internalAB.size - 1, {
        cb.ifx(internalAB.getLength(level) > 0,
          cb += writeInternalNode.invokeCode[Unit](CodeParam(level), CodeParam(false)))
        cb += (level := level + 1)
      })
      cb += (idxOff := utils.invoke[Long]("bytesWritten"))
      cb += writeInternalNode.invokeCode[Unit](CodeParam(level), CodeParam(true))
      idxOff.load()
    }
    m
  }

  def add(cb: EmitCodeBuilder, key: => IEmitCode, offset: Code[Long], annotation: => IEmitCode) {
    cb.ifx(leafBuilder.ab.length.ceq(branchingFactor),
      cb += writeLeafNode.invokeCode[Unit]())
    leafBuilder.add(cb, key, offset, annotation)
    cb += (elementIdx := elementIdx + 1L)
  }
  def close(cb: EmitCodeBuilder): Unit = {
    val off = cb.newLocal[Long]("lastOffset")
    cb += (off := flush.invokeCode[Long]())
    cb += utils.invoke[Unit]("close")
    leafBuilder.close(cb)
    cb += internalAB.close()
    cb += utils.invoke[StagedIndexMetadata, Int, Long, Long, Unit]("writeMetadata", metadata, internalAB.size + 1, off, elementIdx)
  }

  def init(cb: EmitCodeBuilder, path: Value[String]): Unit = {
    val internalBuilder = new StagedInternalNodeBuilder(branchingFactor, keyType, annotationType, cb.localBuilder)
    cb += (elementIdx := 0L)
    cb += (utils := Code.newInstance[IndexWriterUtils, String, FS](path.get, cb.emb.getFS))
    cb += (metaOS := cb.emb.create(path.concat("/metadata.json.gz")))
    cb += (ob := IndexWriter.spec.buildCodeOutputBuffer(utils.invoke[OutputStream]("os")))
    cb += internalAB.create(8)
    leafBuilder.create(cb, 0L)
    internalBuilder.create(cb)
    internalBuilder.store(cb, internalAB, 0)
  }
}