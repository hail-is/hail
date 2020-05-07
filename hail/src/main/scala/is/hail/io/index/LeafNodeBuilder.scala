package is.hail.io.index

import is.hail.annotations.{Annotation, RegionValueBuilder}
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual.{TArray, TInt64, TStruct, Type}
import is.hail.utils.ArrayBuilder

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