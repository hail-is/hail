package is.hail.types.physical

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.backend.HailStateManager
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder}
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.concrete.{SIntervalPointer, SIntervalPointerValue, SStackStruct}
import is.hail.types.physical.stypes.interfaces.primitive
import is.hail.types.physical.stypes.primitives.SBooleanValue
import is.hail.types.virtual.{TInterval, Type}
import is.hail.utils.{FastSeq, Interval}
import org.apache.spark.sql.Row

final case class PCanonicalInterval(pointType: PType, override val required: Boolean = false) extends PInterval {

  override def byteSize: Long = representation.byteSize
  override def alignment: Long = representation.alignment

  override def _asIdent = s"interval_of_${ pointType.asIdent }"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("PCInterval[")
    pointType.pretty(sb, indent, compact)
    sb.append("]")
  }

  val representation: PCanonicalStruct = PCanonicalStruct(
    required,
    "start" -> pointType,
    "end" -> pointType,
    "includesStart" -> PBooleanRequired,
    "includesEnd" -> PBooleanRequired)

  override def setRequired(required: Boolean): PCanonicalInterval =
    if (required == this.required) this else PCanonicalInterval(this.pointType, required)

  override def startOffset(off: Code[Long]): Code[Long] = representation.fieldOffset(off, 0)

  override def endOffset(off: Code[Long]): Code[Long] = representation.fieldOffset(off, 1)

  override def loadStart(off: Long): Long = representation.loadField(off, 0)

  override def loadStart(off: Code[Long]): Code[Long] = representation.loadField(off, 0)

  override def loadEnd(off: Long): Long = representation.loadField(off, 1)

  override def loadEnd(off: Code[Long]): Code[Long] = representation.loadField(off, 1)

  override def startDefined(off: Long): Boolean = representation.isFieldDefined(off, 0)

  override def endDefined(off: Long): Boolean = representation.isFieldDefined(off, 1)

  override def includesStart(off: Long): Boolean = Region.loadBoolean(representation.loadField(off, 2))

  override def includesEnd(off: Long): Boolean = Region.loadBoolean(representation.loadField(off, 3))

  override def startDefined(cb: EmitCodeBuilder, off: Code[Long]): Value[Boolean] =
    representation.isFieldDefined(cb, off, 0)

  override def endDefined(cb: EmitCodeBuilder, off: Code[Long]): Value[Boolean] =
    representation.isFieldDefined(cb, off, 1)

  override def includesStart(off: Code[Long]): Code[Boolean] =
    Region.loadBoolean(representation.loadField(off, 2))

  override def includesEnd(off: Code[Long]): Code[Boolean] =
    Region.loadBoolean(representation.loadField(off, 3))

  override def deepRename(t: Type) = deepRenameInterval(t.asInstanceOf[TInterval])

  private def deepRenameInterval(t: TInterval) =
    PCanonicalInterval(this.pointType.deepRename(t.pointType), this.required)

  override def containsPointers: Boolean = representation.containsPointers

  override def sType: SIntervalPointer = SIntervalPointer(setRequired(false))

  override def loadCheapSCode(cb: EmitCodeBuilder, addr: Code[Long]): SIntervalPointerValue = {
    val a = cb.memoize(addr)
    val incStart = cb.memoize(includesStart(a))
    val incEnd = cb.memoize(includesEnd(a))
    new SIntervalPointerValue(sType, a, incStart, incEnd)
  }

  override def store(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): Value[Long] = {
    value.st match {
      case SIntervalPointer(t: PCanonicalInterval) =>
        representation.store(cb, region, t.representation.loadCheapSCode(cb, value.asInstanceOf[SIntervalPointerValue].a), deepCopy)
      case _ =>
        val interval = value.asInterval
        val start = EmitCode.fromI(cb.emb)(cb => interval.loadStart(cb))
        val stop = EmitCode.fromI(cb.emb)(cb => interval.loadEnd(cb))
        val includesStart = EmitCode.present(cb.emb, new SBooleanValue(interval.includesStart))
        val includesStop = EmitCode.present(cb.emb, new SBooleanValue(interval.includesEnd))
        representation.store(cb, region,
          SStackStruct.constructFromArgs(cb, region, representation.virtualType,
            start, stop, includesStart, includesStop), deepCopy)
    }
  }

  override def storeAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], value: SValue, deepCopy: Boolean): Unit = {
    value.st match {
      case SIntervalPointer(t: PCanonicalInterval) =>
        representation.storeAtAddress(cb, addr, region, t.representation.loadCheapSCode(cb, value.asInstanceOf[SIntervalPointerValue].a), deepCopy)
      case _ =>
        val interval = value.asInterval
        val start = EmitCode.fromI(cb.emb)(cb => interval.loadStart(cb))
        val stop = EmitCode.fromI(cb.emb)(cb => interval.loadEnd(cb))
        val includesStart = EmitCode.present(cb.emb, new SBooleanValue(interval.includesStart))
        val includesStop = EmitCode.present(cb.emb, new SBooleanValue(interval.includesEnd))
        representation.storeAtAddress(cb, addr, region,
          SStackStruct.constructFromArgs(cb, region, representation.virtualType,
            start, stop, includesStart, includesStop),
          deepCopy)
    }
  }

  override def unstagedStoreAtAddress(sm: HailStateManager, addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit = {
    srcPType match {
      case t: PCanonicalInterval =>
        representation.unstagedStoreAtAddress(sm, addr, region, t.representation, srcAddress, deepCopy)
    }
  }

  override def _copyFromAddress(sm: HailStateManager, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long = {
    srcPType match {
      case t: PCanonicalInterval =>
        representation._copyFromAddress(sm, region, t.representation, srcAddress, deepCopy)
    }
  }

  override def loadFromNested(addr: Code[Long]): Code[Long] = representation.loadFromNested(addr)

  override def unstagedLoadFromNested(addr: Long): Long = representation.unstagedLoadFromNested(addr)

  override def unstagedStoreJavaObjectAtAddress(sm: HailStateManager, addr: Long, annotation: Annotation, region: Region): Unit = {
    val jInterval = annotation.asInstanceOf[Interval]
    representation.unstagedStoreJavaObjectAtAddress(
      sm,
      addr,
      Row(jInterval.start, jInterval.end, jInterval.includesStart, jInterval.includesEnd),
      region
    )
  }

  override def unstagedStoreJavaObject(sm: HailStateManager, annotation: Annotation, region: Region): Long = {
    val addr = representation.allocate(region)
    unstagedStoreJavaObjectAtAddress(sm, addr, annotation, region)
    addr
  }

  def constructFromCodes(cb: EmitCodeBuilder, region: Value[Region],
    start: EmitCode, end: EmitCode, includesStart: Value[Boolean], includesEnd: Value[Boolean]
  ): SIntervalPointerValue = {
    val startEC = EmitCode.present(cb.emb, primitive(includesStart))
    val endEC = EmitCode.present(cb.emb, primitive(includesEnd))
    val sc = representation.constructFromFields(cb, region, FastSeq(start, end, startEC, endEC), deepCopy = false)
    new SIntervalPointerValue(sType, sc.a, includesStart, includesEnd)
  }

  override def copiedType: PType = {
    val copiedPoint = pointType.copiedType
    if (copiedPoint.eq(pointType))
      this
    else
      PCanonicalInterval(copiedPoint, required)
  }
}
