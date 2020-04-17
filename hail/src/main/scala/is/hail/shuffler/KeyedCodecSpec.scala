package is.hail.shuffler

import is.hail.expr.types.virtual._
import is.hail.io._

class KeyedCodecSpec(
  t: TStruct,
  codecSpec: TypedCodecSpec,
  key: Array[String]
) {
  val (pType, makeDec) = codecSpec.buildStructDecoder(t)
  val keyPType = pType.selectFields(key)
  val makeEnc = codecSpec.buildEncoder(pType)

  val keyType = t.select(key)._1
  val (decodedKeyPType, makeKeyDec) = codecSpec.buildStructDecoder(keyType)
  val makeKeyEnc = codecSpec.buildEncoder(decodedKeyPType)
}
