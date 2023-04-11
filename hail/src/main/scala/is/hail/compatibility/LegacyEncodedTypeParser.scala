package is.hail.compatibility

import is.hail.expr.ir.IRParser._
import is.hail.expr.ir.{IRParser, PunctuationToken, TokenIterator}
import is.hail.types.encoded._
import is.hail.types.virtual._
import is.hail.rvd.RVDType
import is.hail.utils.FastIndexedSeq

object LegacyEncodedTypeParser {

  def legacy_type_expr(it: TokenIterator): (Type, EType) = {
    val req = it.head match {
      case x: PunctuationToken if x.value == "+" =>
        consumeToken(it)
        true
      case _ => false
    }

    val (vType, eType) = identifier(it) match {
      case "Interval" =>
        punctuation(it, "[")
        val (pointType, ePointType) = legacy_type_expr(it)
        punctuation(it, "]")
        (TInterval(pointType), EBaseStruct(FastIndexedSeq(
          EField("start", ePointType, 0),
          EField("end", ePointType, 1),
          EField("includesStart", EBooleanRequired, 2),
          EField("includesEnd", EBooleanRequired, 3)
        ), req))
      case "Boolean" => (TBoolean, EBoolean(req))
      case "Int32" => (TInt32, EInt32(req))
      case "Int64" => (TInt64, EInt64(req))
      case "Int" => (TInt32, EInt32(req))
      case "Float32" => (TFloat32, EFloat32(req))
      case "Float64" => (TFloat64, EFloat64(req))
      case "String" => (TString, EBinary(req))
      case "Locus" =>
        punctuation(it, "(")
        val rg = identifier(it)
        punctuation(it, ")")
        (TLocus(rg), EBaseStruct(FastIndexedSeq(
          EField("contig", EBinaryRequired, 0),
          EField("position", EInt32Required, 1)), req))
      case "Call" => (TCall, EInt32(req))
      case "Array" =>
        punctuation(it, "[")
        val (elementType, elementEType) = legacy_type_expr(it)
        punctuation(it, "]")
        (TArray(elementType), EArray(elementEType, req))
      case "Set" =>
        punctuation(it, "[")
        val (elementType, elementEType) = legacy_type_expr(it)
        punctuation(it, "]")
        (TSet(elementType), EArray(elementEType, req))
      case "Dict" =>
        punctuation(it, "[")
        val (keyType, keyEType) = legacy_type_expr(it)
        punctuation(it, ",")
        val (valueType, valueEType) = legacy_type_expr(it)
        punctuation(it, "]")
        (TDict(keyType, valueType), EArray(EBaseStruct(FastIndexedSeq(
          EField("key", keyEType, 0),
          EField("value", valueEType, 1)), required = true),
          req))
      case "Tuple" =>
        punctuation(it, "[")
        val types = repsepUntil(it, legacy_type_expr, PunctuationToken(","), PunctuationToken("]"))
        punctuation(it, "]")
        (TTuple(types.map(_._1): _*), EBaseStruct(types.zipWithIndex.map { case ((_, t), idx) => EField(idx.toString, t, idx) }, req))
      case "Struct" =>
        punctuation(it, "{")
        val args = repsepUntil(it, struct_field(legacy_type_expr), PunctuationToken(","), PunctuationToken("}"))
        punctuation(it, "}")
        val (vFields, eFields) = args.zipWithIndex.map { case ((id, (vt, et)), i) => (Field(id, vt, i), EField(id, et, i)) }.unzip
        (TStruct(vFields), EBaseStruct(eFields, req))
    }
    assert(eType.required == req)
    (vType, eType)
  }

  def rvd_type_expr(it: TokenIterator): LegacyRVDType = {
    identifier(it) match {
      case "RVDType" | "OrderedRVDType" =>
        punctuation(it, "{")
        identifier(it, "key")
        punctuation(it, ":")
        punctuation(it, "[")
        val partitionKey = keys(it)
        val restKey = trailing_keys(it)
        punctuation(it, "]")
        punctuation(it, ",")
        identifier(it, "row")
        punctuation(it, ":")
        val (rowType: TStruct, rowEType) = legacy_type_expr(it)
        LegacyRVDType(rowType, rowEType, partitionKey ++ restKey)
    }
  }


  def parseTypeAndEType(str: String): (Type, EType) = {
    IRParser.parse(str, it => legacy_type_expr(it))
  }

  def parseLegacyRVDType(str: String): LegacyRVDType = IRParser.parse(str, it => rvd_type_expr(it))
}
