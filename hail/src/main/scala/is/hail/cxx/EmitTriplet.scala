package is.hail.cxx

import is.hail.expr.types.physical._
import is.hail.utils.ArrayBuilder

import scala.collection.mutable

object EmitTriplet {
  def apply(pType: PType, setup: Code, m: Code, v: Code, region: EmitRegion): EmitTriplet =
    new ConcreteEmitTriplet(pType, setup, s"($m)", s"($v)", region)
}

abstract class EmitTriplet {
  def pType: PType
  def setup: Code
  def m: Code
  def v: Code
  def region: EmitRegion
  def needsRegion: Boolean = true

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
      mv.toString, vv.toString, region)
  }

  def asEmitStream(sameRegion: Boolean): EmitStream = {
    val fb: FunctionBuilder = region.fb
    val pArray = pType.asInstanceOf[PContainer]
    val arrayRegion = EmitRegion.from(region, sameRegion)

    val a = fb.variable("a", "const char *", v)
    val len = fb.variable("len", "int", pArray.cxxLoadLength(a.toString))
    new EmitStream(pArray, setup, m,
      s"""
         |${ a.define }
         |${ len.define }
         |""".stripMargin, Some(len.toString), region, arrayRegion) {
      val i = fb.variable("i", "int", "0")

      def emit(f: (Code, Code) => Code): Code = {
        s"""
           |for (${ i.define } $i < $len; ++$i) {
           |  ${ arrayRegion.defineIfUsed(sameRegion) }
           |  ${
          f(pArray.cxxIsElementMissing(a.toString, i.toString),
            loadIRIntermediate(pArray.elementType, pArray.cxxElementAddress(a.toString, i.toString)))
        }
           |}
           |""".stripMargin
      }
    }
  }
}

class ConcreteEmitTriplet private(val pType: PType, val setup: Code, val m: Code, val v: Code, val region: EmitRegion) extends EmitTriplet {
  override def needsRegion: Boolean = !pType.isPrimitive || region == null
}

/*
 * A new EmitRegion should get created at the start of every stream (e.g.
 * ArrayRange, for the body of ArrayFlatMap). The rest of the stream is emitted,
 * and if the generated code used to process every element of the stream uses
 * the provided region in the EmitRegion, then the code generation for the node
 * that creates the EmitRegion is responsible for initializing it and acquiring
 * a new region for every element, as well as for adding references from the
 * base region (outside the stream) to the necessary regions within the stream.
 *
 * There is an option in `emitArray` to use the same region throughout the
 * deforested array code if e.g. the array needs to be instantiated, as happens
 * when the array is not ultimately passed to an ArrayFold.
 *
 * The EmitTriplet also carries the region that was used to create the value
 * that it holds; this is used for cases like InsertFields inside the body of an
 * ArrayFold in order to unify the field regions, which could be different.
 */

object EmitRegion {
  def from(parentRegion: EmitRegion, sameRegion: Boolean): EmitRegion =
    if (sameRegion) parentRegion else parentRegion.newRegion()

  def apply(fb: FunctionBuilder, region: Code): EmitRegion = {
    new EmitRegion(fb, region, null)
  }
}

class EmitRegion private (val fb: FunctionBuilder, val baseRegion: Code, _region: Variable) {
  assert(_region == null || _region.typ == "RegionPtr")

  val region: Code = if (_region == null) baseRegion else _region.toString
  override def toString: String = region.toString

  private[this] var isUsed: Boolean = false
  def use(): Unit = { isUsed = true }
  def used: Boolean = isUsed

  def defineIfUsed(sameRegion: Boolean): Code = {
    if (isUsed && !sameRegion && _region != null) _region.define else ""
  }

  def addReference(other: EmitRegion): Code =
    if (other.used && this != other) s"$region->add_reference_to($other);" else ""

  def structBuilder(fb: FunctionBuilder, pType: PBaseStruct): StagedBaseStructTripletBuilder = {
    use(); new StagedBaseStructTripletBuilder(this, fb, pType)
  }

  def arrayBuilder(fb: FunctionBuilder, pType: PContainer): StagedContainerBuilder = {
    use(); new StagedContainerBuilder(fb, region, pType)
  }

  def newRegion(): EmitRegion = new EmitRegion(fb, baseRegion, fb.variable("region", "RegionPtr", s"$baseRegion->get_region()"))
}

object SparkFunctionContext {
  def apply(fb: FunctionBuilder, spark_context: Variable): SparkFunctionContext =
    SparkFunctionContext(s"$spark_context.spark_env_", EmitRegion(fb, s"$spark_context.region_"))

  def apply(fb: FunctionBuilder): SparkFunctionContext = apply(fb, fb.getArg(0))
}

case class SparkFunctionContext(sparkEnv: Code, region: EmitRegion)

abstract class EmitStream(
  val pType: PContainer,
  val setup: Code,
  val m: Code,
  val setupLen: Code,
  val length: Option[Code],
  val region: EmitRegion,
  val arrayRegion: EmitRegion
) extends EmitTriplet {
  type F = (Code, Code) => Code

  def emit(f: F): Code

  val fb: FunctionBuilder = region.fb

  def mapEmit(newPType: PContainer, newArrayRegion: EmitRegion = arrayRegion)(mapper: (F, Code, Code) => Code): EmitStream = {
    val outer = this
    new EmitStream(newPType, setup, m, setupLen, length, region, newArrayRegion) {
      def emit(f: F): Code = outer.emit(mapper(f, _, _))
    }
  }

  def v: Code = {
    length match {
      case Some(l) =>
        val sab = region.arrayBuilder(fb, pType)
        s"""
           |({
           |  ${ setupLen }
           |  ${ sab.start(l) }
           |  ${
          emit { case (xm, xv) =>
            s"""
               |if (${ xm })
               |  ${ sab.setMissing() }
               |else
               |  ${ sab.add(xv) }
               |${ sab.advance() }
               |""".stripMargin
          }
        }
           |  ${ sab.end() };
           |})
           |""".stripMargin

      case None =>
        val xs = fb.variable("xs", s"std::vector<${ typeToCXXType(pType.elementType) }>")
        val ms = fb.variable("ms", "std::vector<bool>")
        val i = fb.variable("i", "int")
        val sab = region.arrayBuilder(fb, pType)
        s"""
           |({
           |  ${ setupLen }
           |  ${ xs.define }
           |  ${ ms.define }
           |  ${
          emit { case (xm, xv) =>
            s"""
               |if (${ xm }) {
               |  $ms.push_back(true);
               |  $xs.push_back(${ typeDefaultValue(pType.elementType) });
               |} else {
               |  $ms.push_back(false);
               |  $xs.push_back($xv);
               |}
               |""".stripMargin
          }
        }
           |  ${ sab.start(s"$xs.size()") }
           |  ${ i.define }
           |  for ($i = 0; $i < $xs.size(); ++$i) {
           |    if ($ms[$i])
           |      ${ sab.setMissing() }
           |   else
           |      ${ sab.add(s"$xs[$i]") }
           |    ${ sab.advance() }
           |  }
           |  ${ sab.end() };
           |})
           |""".stripMargin
    }
  }

  override def asEmitStream(sameRegion: Boolean): EmitStream = {
    assert(sameRegion == (region == arrayRegion))
    this
  }
}