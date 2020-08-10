package is.hail.types.physical
import is.hail.annotations.{CodeOrdering, Region, UnsafeOrdering}
import is.hail.asm4s.{Code, MethodBuilder, Value, TypeInfo, UnitInfo}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.types.virtual.{TVoid, Type}

case object PVoid extends PType with PUnrealizable {
  def virtualType: Type = TVoid

  override val required = true

  def _asIdent = "void"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = sb.append("PVoid")

  def setRequired(required: Boolean) = PVoid

  override def unsafeOrdering(): UnsafeOrdering = throw new NotImplementedError()
}

case object PVoidCode extends PCode with PUnrealizableCode { self =>
  override def typeInfo: TypeInfo[_] = UnitInfo

  override def code: Code[_] = Code._empty

  override def tcode[T](implicit ti: TypeInfo[T]): Code[T] = {
    assert(ti == typeInfo)
    code.asInstanceOf[Code[T]]
  }

  def pt: PType = PVoid

  def memoize(cb: EmitCodeBuilder, name: String): PValue = new PValue {
    val pt = self.pt
    def get: PCode = self
  }
}
