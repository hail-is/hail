package is.hail.types.physical

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, IEmitCode}
import is.hail.types.virtual.{TInterval, Type}
import is.hail.utils.FastIndexedSeq

final case class PCanonicalInterval(pointType: PType, override val required: Boolean = false) extends PInterval {
    def _asIdent = s"interval_of_${pointType.asIdent}"

    override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
      sb.append("PCInterval[")
      pointType.pretty(sb, indent, compact)
      sb.append("]")
    }

    override val representation: PStruct = PCanonicalStruct(
      required,
      "start" -> pointType,
      "end" -> pointType,
      "includesStart" -> PBooleanRequired,
      "includesEnd" -> PBooleanRequired)

    def setRequired(required: Boolean) = if(required == this.required) this else PCanonicalInterval(this.pointType, required)

    def startOffset(off: Code[Long]): Code[Long] = representation.fieldOffset(off, 0)

    def endOffset(off: Code[Long]): Code[Long] = representation.fieldOffset(off, 1)

    def loadStart(off: Long): Long = representation.loadField(off, 0)

    def loadStart(off: Code[Long]): Code[Long] = representation.loadField(off, 0)

    def loadEnd(off: Long): Long = representation.loadField(off, 1)

    def loadEnd(off: Code[Long]): Code[Long] = representation.loadField(off, 1)

    def startDefined(off: Long): Boolean = representation.isFieldDefined(off, 0)

    def endDefined(off: Long): Boolean = representation.isFieldDefined(off, 1)

    def includesStart(off: Long): Boolean = Region.loadBoolean(representation.loadField(off, 2))

    def includesEnd(off: Long): Boolean = Region.loadBoolean(representation.loadField(off, 3))

    def startDefined(off: Code[Long])(implicit line: LineNumber): Code[Boolean] =
      representation.isFieldDefined(off, 0)

    def endDefined(off: Code[Long])(implicit line: LineNumber): Code[Boolean] =
      representation.isFieldDefined(off, 1)

    def includesStart(off: Code[Long])(implicit line: LineNumber): Code[Boolean] =
      Region.loadBoolean(representation.loadField(off, 2))

    def includesEnd(off: Code[Long])(implicit line: LineNumber): Code[Boolean] =
      Region.loadBoolean(representation.loadField(off, 3))

    override def deepRename(t: Type) = deepRenameInterval(t.asInstanceOf[TInterval])

    private def deepRenameInterval(t: TInterval) =
      PCanonicalInterval(this.pointType.deepRename(t.pointType),  this.required)
}

object PCanonicalIntervalSettable {
  def apply(sb: SettableBuilder, pt: PCanonicalInterval, name: String): PCanonicalIntervalSettable = {
    new PCanonicalIntervalSettable(pt,
      sb.newSettable[Long](s"${ name }_a"),
      sb.newSettable[Boolean](s"${ name }_includes_start"),
      sb.newSettable[Boolean](s"${ name }_includes_end"))
  }
}

class PCanonicalIntervalSettable(
  val pt: PCanonicalInterval,
  a: Settable[Long],
  val includesStart: Settable[Boolean],
  val includesEnd: Settable[Boolean]
) extends PIntervalValue with PSettable {
  def get(implicit line: LineNumber): PIntervalCode =
    new PCanonicalIntervalCode(pt, a)

  def settableTuple(): IndexedSeq[Settable[_]] = FastIndexedSeq(a, includesStart, includesEnd)

  def loadStart(cb: EmitCodeBuilder)(implicit line: LineNumber): IEmitCode =
    IEmitCode(cb,
      !(pt.startDefined(a)),
      pt.pointType.load(pt.startOffset(a)))

  def loadEnd(cb: EmitCodeBuilder)(implicit line: LineNumber): IEmitCode =
    IEmitCode(cb,
      !(pt.endDefined(a)),
      pt.pointType.load(pt.endOffset(a)))

  def store(pc: PCode)(implicit line: LineNumber): Code[Unit] = {
    Code(
      a := pc.asInstanceOf[PCanonicalIntervalCode].a,
      includesStart := pt.includesStart(a.load()),
      includesEnd := pt.includesEnd(a.load()))
  }
}

class PCanonicalIntervalCode(val pt: PCanonicalInterval, val a: Code[Long]) extends PIntervalCode {
  def code: Code[_] = a

  def codeTuple(): IndexedSeq[Code[_]] = FastIndexedSeq(a)

  def includesStart()(implicit line: LineNumber): Code[Boolean] =
    pt.includesStart(a)

  def includesEnd()(implicit line: LineNumber): Code[Boolean] =
    pt.includesEnd(a)

  def memoize(cb: EmitCodeBuilder, name: String, sb: SettableBuilder)(implicit line: LineNumber): PIntervalValue = {
    val s = PCanonicalIntervalSettable(sb, pt, name)
    cb.assign(s, this)
    s
  }

  def memoize(cb: EmitCodeBuilder, name: String)(implicit line: LineNumber): PIntervalValue =
    memoize(cb, name, cb.localBuilder)

  def memoizeField(cb: EmitCodeBuilder, name: String)(implicit line: LineNumber): PIntervalValue =
    memoize(cb, name, cb.fieldBuilder)

  def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long]): Code[Unit] =
    pt.constructAtAddress(mb, dst, r, pt, a, deepCopy = false)
}
