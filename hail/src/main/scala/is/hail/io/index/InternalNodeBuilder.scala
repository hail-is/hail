package is.hail.io.index

import is.hail.annotations.Region
import is.hail.asm4s.{Code, SettableBuilder, Value}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.OutputBuffer
import is.hail.types.encoded.EType
import is.hail.types.physical._
import is.hail.types.physical.stypes.concrete.{
  SBaseStructPointer, SBaseStructPointerSettable, SIndexablePointerValue,
}
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.virtual.{TStruct, Type}

object InternalNodeBuilder {
  def virtualType(keyType: Type, annotationType: Type): TStruct =
    typ(PType.canonical(keyType), PType.canonical(annotationType)).virtualType

  def legacyTyp(keyType: PType, annotationType: PType) = PCanonicalStruct(
    "children" -> +PCanonicalArray(
      +PCanonicalStruct(
        "index_file_offset" -> +PInt64(),
        "first_idx" -> +PInt64(),
        "first_key" -> keyType,
        "first_record_offset" -> +PInt64(),
        "first_annotation" -> annotationType,
      ),
      required = true,
    )
  )

  def arrayType(keyType: PType, annotationType: PType) =
    PCanonicalArray(
      PCanonicalStruct(
        required = true,
        "index_file_offset" -> +PInt64(),
        "first_idx" -> +PInt64(),
        "first_key" -> keyType,
        "first_record_offset" -> +PInt64(),
        "first_annotation" -> annotationType,
      ),
      required = true,
    )

  def typ(keyType: PType, annotationType: PType) = PCanonicalStruct(
    "children" -> arrayType(keyType, annotationType)
  )
}

class StagedInternalNodeBuilder(
  maxSize: Int,
  keyType: PType,
  annotationType: PType,
  sb: SettableBuilder,
) {
  private val region = sb.newSettable[Region]("internal_node_region")

  val ab = new IndexWriterArrayBuilder(
    "internal_node",
    maxSize,
    sb,
    region,
    InternalNodeBuilder.arrayType(keyType, annotationType),
  )

  val pType: PCanonicalStruct = InternalNodeBuilder.typ(keyType, annotationType)

  private val node = new SBaseStructPointerSettable(
    SBaseStructPointer(pType),
    sb.newSettable[Long]("internal_node_node"),
  )

  def loadFrom(cb: EmitCodeBuilder, ib: StagedIndexWriterUtils, idx: Value[Int]): Unit = {
    cb.assign(region, ib.getRegion(idx))
    cb.assign(node.a, ib.getArrayOffset(idx))
    val aoff = node.loadField(cb, 0).get(cb).asInstanceOf[SIndexablePointerValue].a
    ab.loadFrom(cb, aoff, ib.getLength(idx))
  }

  def store(cb: EmitCodeBuilder, ib: StagedIndexWriterUtils, idx: Value[Int]): Unit =
    ib.update(cb, idx, region.get, node.a.get, ab.length)

  def reset(cb: EmitCodeBuilder): Unit = {
    cb += region.invoke[Unit]("clear")
    allocate(cb)
  }

  def allocate(cb: EmitCodeBuilder): Unit = {
    node.store(cb, pType.loadCheapSCode(cb, pType.allocate(region)))
    ab.create(cb, pType.fieldOffset(node.a, "children"))
  }

  def create(cb: EmitCodeBuilder): Unit = {
    cb.assign(region, Region.stagedCreate(Region.REGULAR, cb.emb.ecb.pool()))
    allocate(cb)
  }

  def encode(cb: EmitCodeBuilder, ob: Value[OutputBuffer]): Unit = {
    val enc = EType.defaultFromPType(pType).buildEncoder(SBaseStructPointer(pType), cb.emb.ecb)
    ab.storeLength(cb)
    enc(cb, node, ob)
  }

  def nodeAddress: SBaseStructValue = node

  def add(
    cb: EmitCodeBuilder,
    indexFileOffset: Code[Long],
    firstIndex: Code[Long],
    firstChild: SBaseStructValue,
  ): Unit = {
    ab.addChild(cb)
    ab.setFieldValue(cb, "index_file_offset", primitive(cb.memoize(indexFileOffset)))
    ab.setFieldValue(cb, "first_idx", primitive(cb.memoize(firstIndex)))
    ab.setField(cb, "first_key", firstChild.loadField(cb, "key"))
    ab.setField(cb, "first_record_offset", firstChild.loadField(cb, "offset"))
    ab.setField(cb, "first_annotation", firstChild.loadField(cb, "annotation"))
  }

  def add(cb: EmitCodeBuilder, indexFileOffset: Code[Long], firstChild: SBaseStructValue): Unit = {
    ab.addChild(cb)
    ab.setFieldValue(cb, "index_file_offset", primitive(cb.memoize(indexFileOffset)))
    ab.setField(cb, "first_idx", firstChild.loadField(cb, "first_idx"))
    ab.setField(cb, "first_key", firstChild.loadField(cb, "first_key"))
    ab.setField(cb, "first_record_offset", firstChild.loadField(cb, "first_record_offset"))
    ab.setField(cb, "first_annotation", firstChild.loadField(cb, "first_annotation"))
  }

  def loadChild(cb: EmitCodeBuilder, idx: Code[Int]): Unit = ab.loadChild(cb, idx)
  def getLoadedChild: SBaseStructValue = ab.getLoadedChild
}
