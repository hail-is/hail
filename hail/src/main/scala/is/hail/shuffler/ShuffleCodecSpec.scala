package is.hail.shuffler

import is.hail.expr.ir._
import is.hail.types.virtual._

class ShuffleCodecSpec(
  ctx: ExecuteContext,
  shuffleType: TShuffle
) {
  val (rowDecodedPType, makeRowDecoder) = shuffleType.rowEType.buildStructDecoder(ctx, shuffleType.rowType)
  assert(rowDecodedPType == shuffleType.rowDecodedPType)
  val makeRowEncoder = shuffleType.rowEType.buildEncoder(ctx, rowDecodedPType)

  val keyType = shuffleType.keyType
  val (keyDecodedPType, makeKeyDecoder) = shuffleType.keyEType.buildStructDecoder(ctx, shuffleType.keyType)
  assert(keyDecodedPType == shuffleType.keyDecodedPType)
  val makeKeyEncoder = shuffleType.keyEType.buildEncoder(ctx, keyDecodedPType)
}
