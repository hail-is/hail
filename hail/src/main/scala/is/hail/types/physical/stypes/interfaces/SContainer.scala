package is.hail.types.physical.stypes.interfaces

import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, IEmitSCode}
import is.hail.types.physical.stypes.{SCode, SType, SValue}

trait SContainer extends SType {
  def elementType: SType
}

trait SIndexableValue extends SValue {
  def loadLength(): Value[Int]

  def isElementMissing(i: Code[Int]): Code[Boolean]

  def isElementDefined(i: Code[Int]): Code[Boolean] = !isElementMissing(i)

  def loadElement(cb: EmitCodeBuilder, i: Code[Int]): IEmitSCode

  def hasMissingValues(cb: EmitCodeBuilder): Code[Boolean]

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
}

trait SIndexableCode extends SCode {
  def st: SContainer

  def loadLength(): Code[Int]

  def memoize(cb: EmitCodeBuilder, name: String): SIndexableValue

  def memoizeField(cb: EmitCodeBuilder, name: String): SIndexableValue
}

