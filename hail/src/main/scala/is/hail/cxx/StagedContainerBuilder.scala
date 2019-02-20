package is.hail.cxx

import is.hail.annotations.UnsafeUtils
import is.hail.cxx
import is.hail.expr.types.physical.{PContainer, PTuple}

class StagedContainerBuilder(fb: FunctionBuilder, region: Code, containerPType: PContainer) {
  private[this] val aoff = fb.variable("aoff", "char *")
  private[this] val eltOff = fb.variable("eltOff", "char *")
  private[this] val i = fb.variable("i", "int", "0")
  private[this] val eltRequired = containerPType.elementType.required

  def start(len: Code, clearMissing: Boolean = true): Code = {
    val __len = fb.variable("len", "int", len)
    s"""${ __len.define }
       |${ start(__len, clearMissing) }
     """.stripMargin
  }

  def start(len: Variable, clearMissing: Boolean): Code = {
    val nMissingBytes = containerPType.cxxNMissingBytes(len.toString)
    val elementsOffset = containerPType.cxxElementsOffset(len.toString)
    val allocate = s"${ region }->allocate(${ containerPType.contentsAlignment }, ${ containerPType.cxxContentsByteSize(len.toString) })"

    s"""${ aoff.defineWith(allocate) }
       |${ eltOff.defineWith(s"$aoff + $elementsOffset") }
       |${ i.define }
       |store_int($aoff, ${ len });
       |${ if (!eltRequired && clearMissing) s"memset($aoff + 4, 0, $nMissingBytes);" else "" }
     """.stripMargin

  }

  def add(x: Code): Code = {
    s"${
      storeIRIntermediate(containerPType.elementType,
        eltOffset,
        x)
    };"
  }

  def setMissing(): Code = {
    if (eltRequired)
      fb.nativeError("Required array element cannot be missing.")
    else
      s"set_bit($aoff + 4, $i);"
  }

  def advance(): Code = s"++$i;"

  def idx: Code = i.toString

  def eltOffset: Code = s"$eltOff + $i * ${ UnsafeUtils.arrayElementSize(containerPType.elementType) }"

  def end(): Code = aoff.toString

  def asClass(tub: TranslationUnitBuilder, wrappedType: PTuple = null): Class = {
    val c = tub.buildClass(tub.genSym("ArrayBuilder"))

    c += s"int len_;"
    c += s"Region * region_;"
    c += s"char * offset_;"


    val nMissingBytes = containerPType.cxxNMissingBytes("len_")
    val eltsOffset = s"offset_ + ${ containerPType.cxxElementsOffset("len_") } + i * ${ UnsafeUtils.arrayElementSize(containerPType.elementType) }"
    c +=
      s"""
         |${c.name}(Region * region, int len) :
         |len_(len),
         |region_(region),
         |offset_(region->allocate(${ containerPType.contentsAlignment }, ${ containerPType.cxxContentsByteSize("len") })) {
         |  store_int(offset_, len_);
         |  ${ if (!eltRequired) s"memset(offset_ + 4, 0, $nMissingBytes);" else "" }
         |}
       """.stripMargin

    if (wrappedType == null) {
      c +=
        s"""
           |void set_element(int i, ${ typeToCXXType(containerPType.elementType) } elt) {
           |  ${ storeIRIntermediate(containerPType.elementType, eltsOffset, "elt") };
           |}
         """.stripMargin
    } else {
      assert(wrappedType.types(0) == containerPType.elementType)
      c +=
      s"""
         |void set_element(int i, ${ typeToCXXType(wrappedType) } elt) {
         |  ${ storeIRIntermediate(containerPType.elementType, eltsOffset, wrappedType.cxxLoadField("elt", 0)) };
         |}
       """.stripMargin

    }

    c += s"char * offset() { return offset_; }"

    c.end()
  }
}
