package is.hail.io.index

import is.hail.annotations.{Annotation, Region, RegionPool, RegionValueBuilder}
import is.hail.asm4s.{HailClassLoader, _}
import is.hail.backend.{ExecuteContext, HailStateManager, HailTaskContext}
import is.hail.expr.ir.{CodeParam, EmitClassBuilder, EmitCodeBuilder, EmitFunctionBuilder, EmitMethodBuilder, IEmitCode, IntArrayBuilder, LongArrayBuilder, ParamType}
import is.hail.io._
import is.hail.io.fs.FS
import is.hail.rvd.AbstractRVDSpec
import is.hail.types
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.concrete.{SBaseStructPointer, SBaseStructPointerSettable}
import is.hail.types.physical.stypes.interfaces.SBaseStructValue
import is.hail.types.physical.{PCanonicalArray, PCanonicalStruct, PType}
import is.hail.types.virtual.Type
import is.hail.utils._
import is.hail.utils.richUtils.ByteTrackingOutputStream
import org.json4s.jackson.Serialization

import java.io.OutputStream

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

  def toFileMetadata: VariableMetadata = VariableMetadata(
    branchingFactor, height, nKeys, rootOffset, attributes
  )
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

object IndexWriter {
  val version: SemanticVersion = SemanticVersion(1, 2, 0)

  val spec: BufferSpec = BufferSpec.default
  def builder(
    ctx: ExecuteContext,
    keyType: PType,
    annotationType: PType,
    branchingFactor: Int = 4096,
    attributes: Map[String, Any] = Map.empty[String, Any]
  ): (String, HailClassLoader, HailTaskContext, RegionPool) => IndexWriter = {
      val sm = ctx.stateManager;
      val f = StagedIndexWriter.build(ctx, keyType, annotationType, branchingFactor);
    { (path: String, hcl: HailClassLoader, htc: HailTaskContext, pool: RegionPool) =>
      new IndexWriter(sm, keyType, annotationType, f(path, hcl, htc, pool, attributes), pool, attributes)
    }
  }
}

class IndexWriter(sm: HailStateManager, keyType: PType, valueType: PType, comp: CompiledIndexWriter, pool: RegionPool, attributes: Map[String, Any]) extends AutoCloseable {
  private val region = Region(pool=pool)
  private val rvb = new RegionValueBuilder(sm, region)
  def appendRow(x: Annotation, offset: Long, annotation: Annotation): Unit = {
    val koff = keyType.unstagedStoreJavaObject(sm, x, region)
    val voff = valueType.unstagedStoreJavaObject(sm, annotation, region)
    comp.apply(koff, offset, voff)
  }

  def trackedOS(): ByteTrackingOutputStream = comp.trackedOS()

  def close(): Unit = {
    region.close()
    comp.close()
  }
}

class IndexWriterArrayBuilder(name: String, maxSize: Int, sb: SettableBuilder, region: Value[Region], arrayType: PCanonicalArray) {
  private val aoff = sb.newSettable[Long](s"${name}_aoff")
  private val len = sb.newSettable[Int](s"${name}_len")

  val eltType: PCanonicalStruct = types.tcoerce[PCanonicalStruct](arrayType.elementType.setRequired((false)))
  private val elt = new SBaseStructPointerSettable(SBaseStructPointer(eltType), sb.newSettable[Long](s"${name}_elt_off"))

  def length: Code[Int] = len

  def loadFrom(cb: EmitCodeBuilder, a: Code[Long], l: Code[Int]): Unit = {
    cb.assign(aoff, a)
    cb.assign(len, l)
  }

  def create(cb: EmitCodeBuilder, dest: Code[Long]): Unit = {
    cb.assign(aoff, arrayType.allocate(region, maxSize))
    arrayType.stagedInitialize(cb, aoff, maxSize)
    arrayType.storeAtAddress(cb, dest, region, arrayType.loadCheapSCode(cb, aoff), deepCopy = false)
    cb.assign(len, 0)
  }

  def storeLength(cb: EmitCodeBuilder): Unit = arrayType.storeLength(cb, aoff, length)

  def setFieldValue(cb: EmitCodeBuilder, name: String, field: SValue): Unit = {
    eltType.setFieldPresent(cb, elt.a, name)
    eltType.fieldType(name).storeAtAddress(cb, eltType.fieldOffset(elt.a, name), region, field, deepCopy = true)
  }

  def setField(cb: EmitCodeBuilder, name: String, v: => IEmitCode): Unit =
    v.consume(cb,
      eltType.setFieldMissing(cb, elt.a, name),
      sv => setFieldValue(cb, name, sv))

  def addChild(cb: EmitCodeBuilder): Unit = {
    loadChild(cb, len)
    cb.assign(len, len + 1)
  }
  def loadChild(cb: EmitCodeBuilder, idx: Code[Int]): Unit = elt.store(cb, eltType.loadCheapSCode(cb, arrayType.loadElement(aoff, idx)))
  def getLoadedChild: SBaseStructValue = elt
}

class StagedIndexWriterUtils(ib: Settable[IndexWriterUtils]) {
  def create(cb: EmitCodeBuilder, path: Code[String], fs: Code[FS], meta: Code[StagedIndexMetadata]): Unit =
    cb.assign(ib, Code.newInstance[IndexWriterUtils, String, FS, StagedIndexMetadata](path, fs, meta))
  def size: Code[Int] = ib.invoke[Int]("size")
  def add(cb: EmitCodeBuilder, r: Code[Region], aoff: Code[Long], len: Code[Int]): Unit =
    cb += ib.invoke[Region, Long, Int, Unit]("add", r, aoff, len)

  def update(cb: EmitCodeBuilder, idx: Code[Int], r: Code[Region], aoff: Code[Long], len: Code[Int]): Unit =
    cb += ib.invoke[Int, Region, Long, Int, Unit]("update", idx, r, aoff, len)

  def getRegion(idx: Code[Int]): Code[Region] = ib.invoke[Int, Region]("getRegion", idx)
  def getArrayOffset(idx: Code[Int]): Code[Long] = ib.invoke[Int, Long]("getArrayOffset", idx)
  def getLength(idx: Code[Int]): Code[Int] = ib.invoke[Int, Int]("getLength", idx)
  def close(cb: EmitCodeBuilder): Unit = cb += ib.invoke[Unit]("close")

  def bytesWritten: Code[Long] = ib.invoke[Long]("bytesWritten")
  def os: Code[OutputStream] = ib.invoke[OutputStream]("os")

  def writeMetadata(cb: EmitCodeBuilder, height: Code[Int], rootOffset: Code[Long], nKeys: Code[Long]): Unit =
    cb += ib.invoke[Int, Long, Long, Unit]("writeMetadata", height, rootOffset, nKeys)
}

case class StagedIndexMetadata(
  branchingFactor: Int,
  keyType: Type,
  annotationType: Type,
  attributes: Map[String, Any]
) {
  def serialize(out: OutputStream, height: Int, rootOffset: Long, nKeys: Long) {
    import AbstractRVDSpec.formats
    val metadata = IndexMetadata(IndexWriter.version.rep, branchingFactor, height, keyType, annotationType, nKeys, "index", rootOffset, attributes)
    Serialization.write(metadata, out)
  }
}

class IndexWriterUtils(path: String, fs: FS, meta: StagedIndexMetadata) {
  val indexPath: String = path + "/index"
  val metadataPath: String = path + "/metadata.json.gz"
  val trackedOS: ByteTrackingOutputStream = new ByteTrackingOutputStream(fs.create(indexPath))

  def bytesWritten: Long = trackedOS.bytesWritten
  def os: OutputStream = trackedOS

  def writeMetadata(height: Int, rootOffset: Long, nKeys: Long): Unit = {
    using(fs.create(metadataPath)) { os => meta.serialize(os, height, rootOffset, nKeys) }
  }

  val rBuilder = new BoxedArrayBuilder[Region]()
  val aBuilder = new LongArrayBuilder()
  val lBuilder = new IntArrayBuilder()

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

  def close(): Unit = {
    rBuilder.result().foreach { r => r.close() }
    trackedOS.close()
  }
}

trait CompiledIndexWriter {
  def init(path: String, attributes: Map[String, Any]): Unit
  def trackedOS(): ByteTrackingOutputStream
  def apply(x: Long, offset: Long, annotation: Long): Unit
  def close(): Unit
}

object StagedIndexWriter {
  def build(
    ctx: ExecuteContext,
    keyType: PType,
    annotationType: PType,
    branchingFactor: Int = 4096
  ): (String, HailClassLoader, HailTaskContext, RegionPool, Map[String, Any]) => CompiledIndexWriter = {
    val fb = EmitFunctionBuilder[CompiledIndexWriter](ctx, "indexwriter",
      FastSeq[ParamType](typeInfo[Long], typeInfo[Long], typeInfo[Long]),
      typeInfo[Unit])
    val cb = fb.ecb
    val siw = new StagedIndexWriter(branchingFactor, keyType, annotationType, cb)

    cb.newEmitMethod("init", FastSeq[ParamType](typeInfo[String], classInfo[Map[String, Any]]), typeInfo[Unit])
      .voidWithBuilder(cb => siw.init(cb, cb.emb.getCodeParam[String](1), cb.emb.getCodeParam[Map[String, Any]](2)))
    fb.emb.voidWithBuilder { cb =>
      siw.add(cb,
        IEmitCode(cb, false, keyType.loadCheapSCode(cb, fb.getCodeParam[Long](1))),
        fb.getCodeParam[Long](2),
        IEmitCode(cb, false, annotationType.loadCheapSCode(cb, fb.getCodeParam[Long](3))))
    }
    cb.newEmitMethod("close", FastSeq[ParamType](), typeInfo[Unit])
      .voidWithBuilder(siw.close)

    cb.newEmitMethod("trackedOS", FastSeq[ParamType](), typeInfo[ByteTrackingOutputStream])
      .emitWithBuilder[ByteTrackingOutputStream] { _ => Code.checkcast[ByteTrackingOutputStream](siw.utils.os) }

    val makeFB = fb.resultWithIndex()

    val fsBc = ctx.fsBc

    { (path: String, hcl: HailClassLoader, htc: HailTaskContext, pool: RegionPool, attributes: Map[String, Any]) =>
      pool.scopedRegion { r =>
        // FIXME: This seems wrong? But also, anywhere we use broadcasting for the FS is wrong.
        val f = makeFB(hcl, fsBc.value, htc, r)
        f.init(path, attributes)
        f
      }
    }
  }

  def withDefaults(keyType: PType, cb: EmitClassBuilder[_],
    branchingFactor: Int = 4096,
    annotationType: PType = +PCanonicalStruct()): StagedIndexWriter =
    new StagedIndexWriter(branchingFactor, keyType, annotationType, cb)
}

class StagedIndexWriter(branchingFactor: Int, keyType: PType, annotationType: PType, cb: EmitClassBuilder[_]) {
  require(branchingFactor > 1)

  private var elementIdx = cb.genFieldThisRef[Long]()
  private val ob = cb.genFieldThisRef[OutputBuffer]()
  private val utils = new StagedIndexWriterUtils(cb.genFieldThisRef[IndexWriterUtils]())

  private val leafBuilder = new StagedLeafNodeBuilder(branchingFactor, keyType, annotationType, cb.fieldBuilder)
  private val writeInternalNode: EmitMethodBuilder[_] =
    cb.defineEmitMethod(genName("m", "writeInternalNode"), FastSeq(IntInfo, BooleanInfo), UnitInfo) { m =>

      val internalBuilder = new StagedInternalNodeBuilder(branchingFactor, keyType, annotationType, m.localBuilder)
      val parentBuilder = new StagedInternalNodeBuilder(branchingFactor, keyType, annotationType, m.localBuilder)

      val level = m.getCodeParam[Int](1)
      val isRoot = m.getCodeParam[Boolean](2)

      m.emitWithBuilder { cb =>
        val idxOff = cb.newLocal[Long]("indexOff")
        cb.assign(idxOff, utils.bytesWritten)
        internalBuilder.loadFrom(cb, utils, level)
        cb += ob.writeByte(1.toByte)
        internalBuilder.encode(cb, ob)
        cb += ob.flush()

      val next = m.newLocal[Int]("next")
      cb.assign(next, level + 1)
      cb.if_(!isRoot, {
        cb.if_(utils.size.ceq(next),
          parentBuilder.create(cb), {
            cb.if_(utils.getLength(next).ceq(branchingFactor),
              cb.invokeVoid(m, cb._this, CodeParam(next), CodeParam(false))
            )
            parentBuilder.loadFrom(cb, utils, next)
          })
        internalBuilder.loadChild(cb, 0)
        parentBuilder.add(cb, idxOff, internalBuilder.getLoadedChild)
        parentBuilder.store(cb, utils, next)
      })

        internalBuilder.reset(cb)
        internalBuilder.store(cb, utils, level)
        Code._empty
      }
    }

  private val writeLeafNode: EmitMethodBuilder[_] =
    cb.defineEmitMethod(genName("m", "writeLeafNode"), FastSeq(), UnitInfo) { m =>
      val parentBuilder = new StagedInternalNodeBuilder(branchingFactor, keyType, annotationType, m.localBuilder)
      m.voidWithBuilder { cb =>
        val idxOff = cb.newLocal[Long]("indexOff")
        cb.assign(idxOff, utils.bytesWritten)
        cb += ob.writeByte(0.toByte)
        leafBuilder.encode(cb, ob)
        cb += ob.flush()

        cb.if_(utils.getLength(0).ceq(branchingFactor),
          cb.invokeVoid(writeInternalNode, cb._this, CodeParam(0), CodeParam(false))
        )
        parentBuilder.loadFrom(cb, utils, 0)

        leafBuilder.loadChild(cb, 0)
        parentBuilder.add(cb, idxOff, leafBuilder.firstIdx(cb).asLong.value, leafBuilder.getLoadedChild)
        parentBuilder.store(cb, utils, 0)
        leafBuilder.reset(cb, elementIdx)
      }
    }

  private val flush: EmitMethodBuilder[_] =
    cb.defineEmitMethod(genName("m", "flush"), FastSeq(), LongInfo) { m =>
      m.emitWithBuilder { cb =>
        val idxOff = cb.newLocal[Long]("indexOff")
        val level = m.newLocal[Int]("level")
        cb.if_(leafBuilder.ab.length > 0, cb.invokeVoid(writeLeafNode, cb._this))
        cb.assign(level, 0)
        cb.while_(level < utils.size - 1, {
          cb.if_(utils.getLength(level) > 0,
            cb.invokeVoid(writeInternalNode, cb._this, CodeParam(level), CodeParam(false))
          )
          cb.assign(level, level + 1)
        })
        cb.assign(idxOff, utils.bytesWritten)
        cb.invokeVoid(writeInternalNode, cb._this, CodeParam(level), CodeParam(true))
        idxOff.load()
      }
    }

  def add(cb: EmitCodeBuilder, key: => IEmitCode, offset: Code[Long], annotation: => IEmitCode) {
    cb.if_(leafBuilder.ab.length.ceq(branchingFactor), cb.invokeVoid(writeLeafNode, cb._this))
    leafBuilder.add(cb, key, offset, annotation)
    cb.assign(elementIdx, elementIdx + 1L)
  }

  def close(cb: EmitCodeBuilder): Unit = {
    val off = cb.invokeCode[Long](flush, cb._this)
    leafBuilder.close(cb)
    utils.close(cb)
    utils.writeMetadata(cb, utils.size + 1, off, elementIdx)
  }

  def init(cb: EmitCodeBuilder, path: Value[String], attributes: Value[Map[String, Any]]): Unit = {
    val metadata = Code.newInstance[StagedIndexMetadata, Int, Type, Type, Map[String, Any]](
      branchingFactor,
      cb.emb.getObject(keyType.virtualType),
      cb.emb.getObject(annotationType.virtualType),
      attributes)
    val internalBuilder = new StagedInternalNodeBuilder(branchingFactor, keyType, annotationType, cb.localBuilder)
    cb.assign(elementIdx, 0L)
    utils.create(cb, path, cb.emb.getFS, metadata)
    cb.assign(ob, IndexWriter.spec.buildCodeOutputBuffer(utils.os))
    leafBuilder.create(cb, 0L)
    internalBuilder.create(cb)
    internalBuilder.store(cb, utils, 0)
  }
}
