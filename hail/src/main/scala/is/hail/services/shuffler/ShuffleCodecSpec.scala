package is.hail.services.shuffler

import is.hail.expr.ir._
import is.hail.types.virtual._
import is.hail.types.physical._
import is.hail.annotations.Region
import org.apache.log4j.Logger

class ShuffleCodecSpec(
  ctx: ExecuteContext,
  shuffleType: TShuffle,
  _rowEncodingPType: Option[PType] = None,
  _keyEncodingPType: Option[PType] = None
) {
  private[this] val log = Logger.getLogger(getClass.getName())

  val (rowDecodedPType, makeRowDecoder) = shuffleType.rowEType.buildStructDecoder(ctx, shuffleType.rowType)
  assert(rowDecodedPType == shuffleType.rowDecodedPType)
  val rowEncodingPType = _rowEncodingPType.getOrElse(rowDecodedPType)
  val makeRowEncoder = shuffleType.rowEType.buildEncoder(ctx, rowEncodingPType)

  val keyType = shuffleType.keyType
  val (keyDecodedPType, makeKeyDecoder) = shuffleType.keyEType.buildStructDecoder(ctx, shuffleType.keyType)
  assert(keyDecodedPType == shuffleType.keyDecodedPType)
  val keyEncodingPType = _keyEncodingPType.getOrElse(keyDecodedPType)
  val makeKeyEncoder = shuffleType.keyEType.buildEncoder(ctx, keyEncodingPType)

  val keyPSubsetStruct = {
    if (keyDecodedPType == rowDecodedPType) {
      rowDecodedPType
    } else {
      new PSubsetStruct(rowDecodedPType, shuffleType.keyFields.map(_.field))
    }
  }
  def constructKeyFromDecodedRow(r: Region, row: Long): Long =
    keyDecodedPType.copyFromAddress(r, keyPSubsetStruct, row, false)

  log.info(s"shuffleType.rowEType: ${shuffleType.rowEType}")
  log.info(s"shuffleType.keyEType: ${shuffleType.keyEType}")

  log.info(s"rowDecodedPType: ${rowDecodedPType}")
  log.info(s"rowEncodingPType: ${rowEncodingPType}")
  log.info(s"keyDecodedPType: ${keyDecodedPType}")
  log.info(s"keyEncodingPType: ${keyEncodingPType}")
}
