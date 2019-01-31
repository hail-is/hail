package is.hail.cxx

import is.hail.expr.types.physical._
import is.hail.utils.ArrayBuilder

object EmitTriplet {
  def apply(pType: PType, setup: Code, m: Code, v: Code): EmitTriplet =
    new EmitTriplet(pType, setup, s"($m)", s"($v)")
}

class EmitTriplet private(val pType: PType, val setup: Code, val m: Code, val v: Code) {
  def memoize(fb: FunctionBuilder): EmitTriplet = {
    val mv = fb.variable("memm", "bool", m)
    val vv = fb.variable("memv", typeToCXXType(pType))

    EmitTriplet(pType,
      s"""
         |$setup
         |${ mv.define }
         |${ vv.define }
         |if (!$mv)
         |  $vv = $v;
         |""".stripMargin,
      mv.toString, vv.toString)
  }
}

class EmitRegion(val region: Variable, init: Code) {
  assert(region.typ == "RegionPtr")

  private[this] var isUsed: Boolean = false

  private[this] var setup: ArrayBuilder[Code] = new ArrayBuilder[Code](16)

  setup += s"$region = $init;"

  def use(): Unit = { isUsed = true }

  def declareIfUsed(): Code = if (isUsed) region.define else ""

  def redefineIfUsed(): Code = if (isUsed) setup.result().mkString("\n") else ""

  def structBuilder(fb: FunctionBuilder, pType: PBaseStruct): StagedBaseStructTripletBuilder = {
    use()
    new StagedBaseStructTripletBuilder(region, fb, pType)
  }

  def arrayBuilder(fb: FunctionBuilder, pType: PContainer): StagedContainerBuilder = {
    use()
    new StagedContainerBuilder(fb, region.name, pType)
  }

  def neededBy(other: EmitRegion): Unit = { setup += s"${ other.region }->add_reference_to($region);" }

  def newDependentRegion(fb: FunctionBuilder): EmitRegion = {
    val newRegion = new EmitRegion(fb.variable("region", "RegionPtr", "nullptr"), s"$region->get_region()")
    newRegion.neededBy(this)
    newRegion
  }
}

abstract class ArrayEmitter(val setup: Code, val m: Code, val setupLen: Code, val length: Option[Code], val arrayRegion: EmitRegion) {
  def emit(f: (Code, Code) => Code): Code
}