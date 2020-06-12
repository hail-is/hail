package is.hail.io.index

import is.hail.annotations.Region
import is.hail.asm4s.{Code, SettableBuilder, Value}
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode}
import is.hail.types.encoded.EType
import is.hail.types.physical._
import is.hail.types.virtual.{TStruct, Type}
import is.hail.io.OutputBuffer
import is.hail.utils._

object LeafNodeBuilder {
  def virtualType(keyType: Type, annotationType: Type): TStruct = typ(PType.canonical(keyType), PType.canonical(annotationType)).virtualType

  def legacyTyp(keyType: PType, annotationType: PType) = PCanonicalStruct(
    "first_idx" -> +PInt64(),
    "keys" -> +PCanonicalArray(+PCanonicalStruct(
      "key" -> keyType,
      "offset" -> +PInt64(),
      "annotation" -> annotationType
    ), required = true))

  def arrayType(keyType: PType, annotationType: PType) =
    PCanonicalArray(PCanonicalStruct(required = true,
      "key" -> keyType,
      "offset" -> +PInt64(),
      "annotation" -> annotationType), required = true)

  def typ(keyType: PType, annotationType: PType) = PCanonicalStruct(
    "first_idx" -> +PInt64(),
    "keys" -> arrayType(keyType, annotationType))
}


class StagedLeafNodeBuilder(maxSize: Int, keyType: PType, annotationType: PType, sb: SettableBuilder) {
  private val region = sb.newSettable[Region]("leaf_node_region")
  val ab = new IndexWriterArrayBuilder("leaf_node", maxSize,
    sb, region,
    LeafNodeBuilder.arrayType(keyType, annotationType))

  private val pType: PCanonicalStruct = LeafNodeBuilder.typ(keyType, annotationType)
  private val node = new PCanonicalBaseStructSettable(pType, sb.newSettable[Long]("lef_node_addr"))

  def close(cb: EmitCodeBuilder): Unit = cb.ifx(!region.isNull, cb += region.invalidate())

  def reset(cb: EmitCodeBuilder, firstIdx: Code[Long]): Unit = {
    cb += region.invoke[Unit]("clear")
    cb += node.store(PCode(pType, pType.allocate(region)))
    cb += PInt64().storePrimitiveAtAddress(pType.fieldOffset(node.a, "first_idx"), PInt64(), firstIdx)
    ab.create(cb, pType.fieldOffset(node.a, "keys"))
  }

  def create(cb: EmitCodeBuilder, firstIdx: Code[Long]): Unit = {
    cb.assign(region, Region.stagedCreate(Region.REGULAR))
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