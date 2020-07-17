package is.hail.services.shuffler

import is.hail.expr.ir._
import is.hail.types.virtual._
import is.hail.types.physical._

class ShuffleCodecSpec(
  ctx: ExecuteContext,
  shuffleType: TShuffle,
  _rowEncodingPType: Option[PType] = None,
  _keyEncodingPType: Option[PType] = None
) {
  val (rowDecodedPType, makeRowDecoder) = shuffleType.rowEType.buildStructDecoder(ctx, shuffleType.rowType)
  assert(rowDecodedPType == shuffleType.rowDecodedPType)
  val rowEncodingPType = _rowEncodingPType.getOrElse(rowDecodedPType)
  val makeRowEncoder = shuffleType.rowEType.buildEncoder(ctx, rowEncodingPType)

  val keyType = shuffleType.keyType
  val (keyDecodedPType, makeKeyDecoder) = shuffleType.keyEType.buildStructDecoder(ctx, shuffleType.keyType)
  assert(keyDecodedPType == shuffleType.keyDecodedPType)
  val keyEncodingPType = _keyEncodingPType.getOrElse(keyDecodedPType)
  val makeKeyEncoder = shuffleType.keyEType.buildEncoder(ctx, keyEncodingPType)
}
