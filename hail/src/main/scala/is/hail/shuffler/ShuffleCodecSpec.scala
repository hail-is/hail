package is.hail.shuffler

import is.hail.expr.ir._
import is.hail.expr.types.encoded._
import is.hail.expr.types.virtual._

class ShuffleCodecSpec(
  ctx: ExecuteContext,
  keyFields: Array[SortField],
  rowType: TStruct,
  rowEType: EBaseStruct,
  keyEType: EBaseStruct
) {
  val (rowDecodedPType, makeRowDecoder) = rowEType.buildStructDecoder(ctx, rowType)
  assert(rowDecodedPType == rowDecodedPType)
  val makeRowEncoder = rowEType.buildEncoder(ctx, rowDecodedPType)

  val keyType = rowType.typeAfterSelectNames(keyFields.map(_.field))
  val (keyDecodedPType, makeKeyDecoder) = keyEType.buildStructDecoder(ctx, keyType)
  assert(keyDecodedPType == keyDecodedPType)
  val makeKeyEncoder = keyEType.buildEncoder(ctx, keyDecodedPType)
}
