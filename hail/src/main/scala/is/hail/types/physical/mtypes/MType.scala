package is.hail.types.physical.mtypes

import is.hail.annotations.Region
import is.hail.asm4s.{Code, CodeLabel, Value}
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode}
import is.hail.types.physical.PCode
import is.hail.types.physical.stypes.{IEmitSCode, SCode, SType, SValue}
import is.hail.utils._

object IEmitMCode {
  def apply(cb: EmitCodeBuilder, m: Code[Boolean], pc: => MCode): IEmitMCode = {
    val Lmissing = CodeLabel()
    val Lpresent = CodeLabel()
    cb.ifx(m, { cb.goto(Lmissing) })
    val resPc: MCode = pc
    cb.goto(Lpresent)
    IEmitMCode(Lmissing, Lpresent, resPc)
  }
}

case class IEmitMCode(Lmissing: CodeLabel, Lpresent: CodeLabel, pc: MCode) {
  def mapS(cb: EmitCodeBuilder)(f: (MCode) => SCode): IEmitSCode = {
    val Lpresent2 = CodeLabel()
    cb.define(Lpresent)
    val sc = f(pc)
    cb.goto(Lpresent2)
    IEmitSCode(Lmissing, Lpresent2, sc)
  }

}

class UninitializedMValue(val addr: Value[Long], val typ: MType) {
  final def store(cb: EmitCodeBuilder, region: Value[Region], code: SCode): MValue = {
    store(cb, region, code.memoize(cb))
  }

  final def store(cb: EmitCodeBuilder, region: Value[Region], value: SValue): MValue = {
    typ.storeFromSValue(cb, this, value)
  }

  protected[mtypes] def toMValue: MValue = new MValue(addr, typ)
}

class MCode(val addr: Code[Long], val typ: MType) {
  def memoize(cb: EmitCodeBuilder): MValue = {
    val x = cb.newLocal[Long]("mcode_memoize")
    cb.assign(x, addr)
    new MValue(x, typ)
  }
}

class MValue(val addr: Value[Long], val typ: MType)


object MType {
  def canonical(st: SType): MType = ???
}

trait MType {
  def byteSize: Long

  def alignment: Long

  def allocate(cb: EmitCodeBuilder, region: Value[Region]): UninitializedMValue = {
    val addr = cb.newLocal[Long]("mtype_alloc")
    cb.assign(addr, region.allocate(alignment, byteSize))
    new UninitializedMValue(addr, this)
  }

  // value should have the same virtual type
  def storeFromSValue(cb: EmitCodeBuilder, memory: UninitializedMValue, value: SValue, region: Value[Region], deepCopy: Boolean): MValue
//

  def pointerType: SType

  // most mtypes are represented inline
  def loadNestedRepr(addr: Code[Long]): Code[Long] = addr
}
