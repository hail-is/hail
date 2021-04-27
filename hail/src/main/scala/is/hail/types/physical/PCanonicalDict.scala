package is.hail.types.physical

import is.hail.annotations.{Annotation, Region}
import is.hail.types.virtual.{TDict, Type}
import is.hail.types.physical.stypes.concrete.{SIndexablePointer, SIndexablePointerCode}
import org.apache.spark.sql.Row

final case class PCanonicalDict(keyType: PType, valueType: PType, required: Boolean = false) extends PDict with PArrayBackedContainer {
  val elementType = PCanonicalStruct(required = true, "key" -> keyType, "value" -> valueType)

  val arrayRep: PCanonicalArray = PCanonicalArray(elementType, required)

  def setRequired(required: Boolean) = if(required == this.required) this else PCanonicalDict(keyType, valueType, required)

  def _asIdent = s"dict_of_${keyType.asIdent}AND${valueType.asIdent}"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("PCDict[")
    keyType.pretty(sb, indent, compact)
    if (compact)
      sb += ','
    else
      sb.append(", ")
    valueType.pretty(sb, indent, compact)
    sb.append("]")
  }

  override def deepRename(t: Type) = deepRenameDict(t.asInstanceOf[TDict])

  private def deepRenameDict(t: TDict) =
    PCanonicalDict(this.keyType.deepRename(t.keyType), this.valueType.deepRename(t.valueType), this.required)

  override def unstagedStoreJavaObject(annotation: Annotation, region: Region): Long = {
    val annotMap = annotation.asInstanceOf[Map[Annotation, Annotation]]
    val sortedArray = annotMap.map{ case (k, v) => Row(k, v) }
      .toArray
      .sorted(elementType.virtualType.ordering.toOrdering)
      .toIndexedSeq
    this.arrayRep.unstagedStoreJavaObject(sortedArray, region)
  }

  def construct(contents: PIndexableCode): PIndexableCode = {
    assert(contents.pt.equalModuloRequired(arrayRep), s"\n  contents:  ${ contents.pt }\n  arrayrep: ${ arrayRep }")
    new SIndexablePointerCode(SIndexablePointer(this), contents.asInstanceOf[SIndexablePointerCode].a)
  }
}
