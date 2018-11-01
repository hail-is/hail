package is.hail.cxx

import is.hail.expr.types.physical.PBaseStruct
import is.hail.utils.ArrayBuilder

class StagedBaseStructBuilder(st: Code, pStruct: PBaseStruct, off: Expression) {

  def setAllMissing(): Code =
    s"memset($off, 0xff, ${ pStruct.nMissingBytes });"

  def clearAllMissing(): Code =
    s"memset($off, 0, ${ pStruct.nMissingBytes });"

  def setMissing(idx: Int): Code = {
    if (pStruct.fieldRequired(idx))
      s"""NATIVE_ERROR($st, 1010, "Required field cannot be missing.");"""
    else
      s"set_bit($off, ${ pStruct.missingIdx(idx) });"
  }

  def clearMissing(idx: Int): Code = {
    if (pStruct.fieldRequired(idx))
      ""
    else
      s"clear_bit($off, ${ pStruct.missingIdx(idx) });"
  }

  def addField(idx: Int, addToOffset: Code => Code): Code =
    addToOffset(s"(($off) + ${ pStruct.byteOffsets(idx) })")

  def offset: Code = off.toString
}

class StagedBaseStructTripletBuilder(st: Code, region: Code, pStruct: PBaseStruct) {
  private[this] val s = Variable("s", "char *", s"$region->allocate(${ pStruct.alignment }, ${ pStruct.byteSize })")
  private[this] val ssb = new StagedBaseStructBuilder(st, pStruct, s.ref)
  private[this] val ab = new ArrayBuilder[Code]
  ab += s.define
  ab += ssb.clearAllMissing()

  private var i = 0

  def body(): Code = {
    assert(i == pStruct.size)
    ab.result().mkString("\n")
  }

  def end(): Code = {
    assert(i == pStruct.size)
    ssb.offset
  }

  def add(t: EmitTriplet) {
    val f = pStruct.fields(i)
    ab +=
      s"""
         |${ t.setup }
         |if (${ t.m }) {
         |  ${ ssb.setMissing(i) }
         |} else {
         |  ${ ssb.addField(i, off => storeIRIntermediate(f.typ, off, t.v)) };
         |}""".stripMargin
    i += 1
  }

  def triplet(): EmitTriplet = EmitTriplet(pStruct, body(), "false", end())
}
