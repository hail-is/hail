package is.hail.types.physical.stypes.interfaces

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode}
import is.hail.types.physical.stypes.primitives.SInt32Code
import is.hail.types.physical.stypes.{EmitType, SCode, SSettable, SType, SValue}
import is.hail.types.{RIterable, TypeWithRequiredness}

trait SContainer extends SType {
  def elementType: SType
  def elementEmitType: EmitType
  override def _typeWithRequiredness: TypeWithRequiredness = RIterable(elementEmitType.typeWithRequiredness.r)
  def constructFromFunctionsKnownLength(cb: EmitCodeBuilder, region: Value[Region], length: Value[Int], deepCopy: Boolean): ((EmitCodeBuilder, IEmitCode) => Unit, EmitCodeBuilder => SIndexableCode)
}

trait SIndexableSettable extends SIndexableValue with SSettable

trait SIndexableValue extends SValue {
  def st: SContainer

  def loadLength(): Value[Int]

  def isElementMissing(cb: EmitCodeBuilder, i: Code[Int]): Code[Boolean]

  def isElementDefined(cb: EmitCodeBuilder, i: Code[Int]): Code[Boolean] = !isElementMissing(cb, i)

  def loadElement(cb: EmitCodeBuilder, i: Code[Int]): IEmitCode

  def hasMissingValues(cb: EmitCodeBuilder): Code[Boolean]

  def numberMissingValues(cb: EmitCodeBuilder): Code[Int] = {
    if (st.elementEmitType.required) {
      const(0)
    } else {
      val (i, c) = cb.newLocal[Int]("i") -> cb.newLocal[Int]("missing_count", 0)
      cb.forLoop(cb.assign(i, 0), i < loadLength(), cb.assign(i, i + 1), {
        cb.assign(c, c + isElementMissing(cb, i).toI)
      })
      c
    }
  }

  def forEachDefined(cb: EmitCodeBuilder)(f: (EmitCodeBuilder, Value[Int], SCode) => Unit): Unit = {
    val length = loadLength()
    val idx = cb.newLocal[Int]("foreach_idx", 0)
    cb.whileLoop(idx < length, {

      loadElement(cb, idx).consume(cb,
        {}, /*do nothing if missing*/
        { eltCode =>
          f(cb, idx, eltCode)
        })
      cb.assign(idx, idx + 1)
    })
  }

  def forEachDefinedOrMissing(cb: EmitCodeBuilder)(missingF: (EmitCodeBuilder, Value[Int]) => Unit, presentF: (EmitCodeBuilder, Value[Int], SCode) => Unit): Unit = {
    val length = loadLength()
    val idx = cb.newLocal[Int]("foreach_idx", 0)
    cb.whileLoop(idx < length, {

      loadElement(cb, idx).consume(cb,
        { /*do function if missing*/
          missingF(cb, idx)
        },
        { eltCode =>
          presentF(cb, idx, eltCode)
        })
      cb.assign(idx, idx + 1)
    })
  }

  override def hash(cb: EmitCodeBuilder): SInt32Code = {
    val hash_result = cb.newLocal[Int]("array_hash", 1)
    forEachDefinedOrMissing(cb)({ case (cb, idx) => cb.assign(hash_result, hash_result * 31) },
      { case (cb, idx, element) => cb.assign(hash_result, hash_result * 31 + element.memoize(cb, "array_hash_element").hash(cb).intCode(cb))
      })
    new SInt32Code(hash_result)
  }
}
trait SIndexableCode extends SCode {
  def st: SContainer

  def codeLoadLength(): Code[Int]

  def memoize(cb: EmitCodeBuilder, name: String): SIndexableValue

  def memoizeField(cb: EmitCodeBuilder, name: String): SIndexableValue

  def castToArray(cb: EmitCodeBuilder): SIndexableCode
}

