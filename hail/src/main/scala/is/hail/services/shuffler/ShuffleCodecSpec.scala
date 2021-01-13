package is.hail.services.shuffler

import is.hail.expr.ir._
import is.hail.types.virtual._
import is.hail.types.physical._
import is.hail.annotations.Region

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

  val keyDecodedSubsetPType = new PSubsetStruct(
    rowDecodedPType,
    shuffleType.keyFields.map(_.field).toArray)
  def constructKeyFromDecodedRow(r: Region, row: Long): Long =
    keyDecodedSubsetPType.copyFromAddress(r, rowDecodedPType, row, false)

  val keyType = shuffleType.keyType
  val (keyDecodedPType, makeKeyDecoder) = shuffleType.keyEType.buildStructDecoder(ctx, shuffleType.keyType)
  assert(keyDecodedPType == shuffleType.keyDecodedPType)
  val keyEncodingPType = _keyEncodingPType.getOrElse(keyDecodedPType)
  val makeKeyEncoder = shuffleType.keyEType.buildEncoder(ctx, keyEncodingPType)
}
