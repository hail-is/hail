package is.hail.utils.richUtils

import is.hail.asm4s.{Code, MethodBuilder, TypeInfo, Value}

class RichCodeIndexedSeq[T](cs: IndexedSeq[Code[T]]) {
  def cacheEntries(mb: MethodBuilder, ti: TypeInfo[T]): (Code[Unit], IndexedSeq[Value[T]]) = {
    val cacheVariables = cs.map(_ => mb.newField(ti))

    val cachingCode = Code.foreach(cacheVariables.zip(cs)) { case (cacheVariable, arrElement) =>
      cacheVariable := arrElement
    }

    (cachingCode, cacheVariables)
  }
}
