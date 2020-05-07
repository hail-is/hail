package is.hail.io.index

import is.hail.annotations.RegionValueBuilder
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual.{TArray, TInt64, TStruct, Type}
import is.hail.utils.ArrayBuilder

object InternalNodeBuilder {
  def virtualType(keyType: Type, annotationType: Type): TStruct = typ(PType.canonical(keyType), PType.canonical(annotationType)).virtualType

  def legacyTyp(keyType: PType, annotationType: PType) = PCanonicalStruct(
    "children" -> +PCanonicalArray(+PCanonicalStruct(
      "index_file_offset" -> +PInt64(),
      "first_idx" -> +PInt64(),
      "first_key" -> keyType,
      "first_record_offset" -> +PInt64(),
      "first_annotation" -> annotationType
    ), required = true)
  )

  def arrayType(keyType: PType, annotationType: PType) =
    PCanonicalArray(PCanonicalStruct(required = true,
      "index_file_offset" -> +PInt64(),
      "first_idx" -> +PInt64(),
      "first_key" -> keyType,
      "first_record_offset" -> +PInt64(),
      "first_annotation" -> annotationType
    ), required = true)

  def typ(keyType: PType, annotationType: PType) = PCanonicalStruct(
    "children" -> arrayType(keyType, annotationType)
  )
}