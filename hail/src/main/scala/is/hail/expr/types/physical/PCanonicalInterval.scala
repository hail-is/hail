package is.hail.expr.types.physical

import is.hail.annotations._
import is.hail.asm4s.Code
import is.hail.expr.types.virtual.{TInterval, Type}

final case class PCanonicalInterval(pointType: PType, override val required: Boolean = false) extends PInterval {
    def _asIdent = s"interval_of_${pointType.asIdent}"

    override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
      sb.append("PCInterval[")
      pointType.pretty(sb, indent, compact)
      sb.append("]")
    }

    override val representation: PStruct = PStruct(
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

    def startDefined(off: Code[Long]): Code[Boolean] = representation.isFieldDefined(off, 0)

    def endDefined(off: Code[Long]): Code[Boolean] = representation.isFieldDefined(off, 1)

    def includeStart(off: Code[Long]): Code[Boolean] =
      Region.loadBoolean(representation.loadField(off, 2))

    def includeEnd(off: Code[Long]): Code[Boolean] =
      Region.loadBoolean(representation.loadField(off, 3))

    override def deepRename(t: Type) = deepRenameInterval(t.asInstanceOf[TInterval])

    private def deepRenameInterval(t: TInterval) =
      PCanonicalInterval(this.pointType.deepRename(t.pointType),  this.required)
}
