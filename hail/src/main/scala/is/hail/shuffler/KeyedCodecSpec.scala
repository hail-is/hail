package is.hail.shuffler

import is.hail.expr.ir.ExecuteContext
import is.hail.types.virtual._
import is.hail.io._

class KeyedCodecSpec(
  ctx: ExecuteContext,
  t: TStruct,
  codecSpec: TypedCodecSpec,
  key: Array[String]
) {
  val (pType, makeDec) = codecSpec.buildStructDecoder(ctx, t)
  val keyPType = pType.selectFields(key)
  val makeEnc = codecSpec.buildEncoder(ctx, pType)

  val keyCodecSpec = TypedCodecSpec(keyPType, codecSpec._bufferSpec)
  val keyType = t.select(key)._1
  val (decodedKeyPType, makeKeyDec) = keyCodecSpec.buildStructDecoder(ctx, keyType)
  val makeKeyEnc = keyCodecSpec.buildEncoder(ctx, decodedKeyPType)
}
