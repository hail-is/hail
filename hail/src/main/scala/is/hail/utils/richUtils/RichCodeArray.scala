package is.hail.utils.richUtils

import is.hail.asm4s.{Code, MethodBuilder, TypeInfo}

class RichCodeArray[T](arr: Array[Code[T]]) {
  def cacheEntries(mb: MethodBuilder, ti: TypeInfo[T]): (Code[Unit], Array[Code[T]]) = {
    val cacheVariables = arr.map(_ => mb.newField(ti))

    val cachingCode = Code(
      cacheVariables.zip(arr).map { case (cacheVariable, arrElement) =>
        cacheVariable := arrElement
      }:_*
    ).asInstanceOf[Code[Unit]]

    (cachingCode, cacheVariables.map(_.load()))
  }
}
