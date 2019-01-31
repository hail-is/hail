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

/*
 * A new EmitRegion should get created at the start of every stream (e.g.
 * ArrayRange, for the body of ArrayFlatMap). The rest of the stream is emitted,
 * and if the generated code used to process every element of the stream uses
 * the provided region in the EmitRegion, then the code generation for the node
 * that creates the EmitRegion is responsible for initializing it and acquiring
 * a new region for every element, as well as for adding references from the base
 * region (outside the stream) to the necessary regions within the stream.
 *
 * This would be better if:
 *  - The (stream) region wasn't added to the base region if unneeded,
 *    regardless of whether the computation used the region (e.g. computation
 *    produces an integer, but creates intermediary structs/arrays)
 *  - Instead of needing to instantiate a stream region for and then reference
 *    it in the base region, we should just have those things compute directly
 *    into the base region.
 *  - In addition to that, we can create scratch regions e.g. for filter
 *    conditions, so that values that don't appear in the final result can be
 *    released as long as they are no longer useful.
 */

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