package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.collection.implicits.{toRichIterable, RichIterable}
import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.Pretty.prettyBooleanLiteral
import is.hail.expr.ir.agg._
import is.hail.expr.ir.defs._
import is.hail.expr.ir.functions.RelationalFunctions
import is.hail.types.virtual.{MatrixType, TArray, TInterval, TStream, TableType, Type}
import is.hail.utils.{space => _, _}
import is.hail.utils.prettyPrint._

import scala.annotation.switch
import scala.collection.immutable.ArraySeq
import scala.collection.mutable

import org.json4s.DefaultFormats
import org.json4s.jackson.{JsonMethods, Serialization}

object Pretty {
  def apply(
    ctx: ExecuteContext,
    ir: BaseIR,
    width: Int = 100,
    ribbonWidth: Int = 50,
    elideLiterals: Boolean = true,
    maxLen: Int = -1,
    allowUnboundRefs: Boolean = false,
    preserveNames: Boolean = true,
  ): String = {
    val useSSA = ctx != null && ctx.getFlag("use_ssa_logs") != null
    val pretty =
      new Pretty(width, ribbonWidth, elideLiterals, maxLen, allowUnboundRefs, useSSA, preserveNames)
    pretty(ir)
  }

  def sexprStyle(
    ir: BaseIR,
    width: Int = 100,
    ribbonWidth: Int = 50,
    elideLiterals: Boolean = true,
    maxLen: Int = -1,
    allowUnboundRefs: Boolean = false,
  ): String = {
    val pretty =
      new Pretty(width, ribbonWidth, elideLiterals, maxLen, allowUnboundRefs, useSSA = false)
    pretty(ir)
  }

  def ssaStyle(
    ir: BaseIR,
    width: Int = 100,
    ribbonWidth: Int = 50,
    elideLiterals: Boolean = true,
    maxLen: Int = -1,
    allowUnboundRefs: Boolean = false,
    preserveNames: Boolean = true,
  ): String = {
    val pretty =
      new Pretty(width, ribbonWidth, elideLiterals, maxLen, allowUnboundRefs, useSSA = true,
        preserveNames = preserveNames)
    pretty(ir)
  }

  def prettyBooleanLiteral(b: Boolean): String =
    if (b) "True" else "False"

  def prettyClass(x: AnyRef): String = {
    val name = x.getClass.getName.split("\\.").last
    if (name.endsWith("$")) name.substring(0, name.length - 1)
    else name
  }
}

class Pretty(
  width: Int,
  ribbonWidth: Int,
  elideLiterals: Boolean,
  maxLen: Int,
  allowUnboundRefs: Boolean,
  useSSA: Boolean,
  preserveNames: Boolean = false,
) {
  def short(ir: BaseIR): String = {
    val s = apply(ir)
    if (s.length < maxLen) s else s.substring(0, maxLen) + "..."
  }

  def prettyStringLiteral(s: String, elide: Boolean = false, maxLen: Int = 1000): String = {
    val esc = StringEscapeUtils.escapeString(s)
    if (elide && esc.length > maxLen) {
      s""""${esc.substring(0, maxLen)}...""""
    } else {
      s""""$esc""""
    }
  }

  def fillList(docs: Iterable[Doc], indent: Int = 2): Doc =
    group(lineAlt, nest(indent, prettyPrint.fillList(docs)))

  def fillList(docs: Doc*): Doc = fillList(docs)

  def list(docs: Iterable[Doc], indent: Int = 2): Doc =
    nest(indent, prettyPrint.list(docs))

  def list(docs: Doc*): Doc = list(docs)

  def prettyStrings(xs: IndexedSeq[String]): Doc =
    fillList(xs.view.map(x => text(prettyStringLiteral(x))))

  def prettyStringsOpt(x: Option[IndexedSeq[String]]): Doc =
    x.map(prettyStrings).getOrElse("None")

  def prettyTypes(x: Seq[Type]): Doc =
    fillList(x.view.map(typ => text(typ.parsableString())))

  def prettySortFields(x: Seq[SortField]): Doc =
    fillList(x.view.map(typ => text(typ.parsableString())))

  def prettySortFieldsString(x: Seq[SortField]): String =
    x.view.map(_.parsableString()).mkString("(", " ", ")")

  val MAX_VALUES_TO_LOG: Int = 25

  def prettyIntOpt(x: Option[Int]): String =
    x.map(_.toString).getOrElse("None")

  def prettyLong(x: Long): Doc =
    text(x.toString)

  def prettyLongs(x: IndexedSeq[Long], elideLiterals: Boolean): Doc = {
    val truncate = elideLiterals && x.length > MAX_VALUES_TO_LOG
    val view = if (truncate) x.view else x.view.slice(0, MAX_VALUES_TO_LOG)
    val docs = view.map(prettyLong)
    concat(docs.intersperse[Doc](
      "(",
      softline,
      if (truncate) s"... ${x.length - MAX_VALUES_TO_LOG} more values... )" else ")",
    ))
  }

  def prettyInt(x: Int): Doc =
    text(x.toString)

  def prettyInts(x: IndexedSeq[Int], elideLiterals: Boolean): Doc = {
    val truncate = elideLiterals && x.length > MAX_VALUES_TO_LOG
    val view = if (truncate) x.view else x.view.slice(0, MAX_VALUES_TO_LOG)
    val docs = view.map(prettyInt)
    concat(docs.intersperse[Doc](
      "(",
      softline,
      if (truncate) s"... ${x.length - MAX_VALUES_TO_LOG} more values... )" else ")",
    ))
  }

  def prettyIdentifiers(x: IndexedSeq[String]): Doc =
    fillList(x.view.map(text))

  def prettyName(name: Name): Doc = text(name.str)

  def prettyNames(x: IndexedSeq[Name]): Doc =
    fillList(x.view.map(prettyName))

  def prettyAggStateSignatures(states: Seq[AggStateSig]): Doc =
    list(states.view.map(prettyAggStateSignature))

  def prettyAggStateSignature(state: AggStateSig): Doc = {
    state match {
      case FoldStateSig(resultEmitType, accumName, otherAccumName, combOpIR) =>
        fillList(IndexedSeq(
          text(Pretty.prettyClass(state)),
          text(resultEmitType.typeWithRequiredness.canonicalPType.toString),
          prettyName(accumName),
          prettyName(otherAccumName),
          text(apply(combOpIR)),
        ))
      case _ =>
        fillList(state.n match {
          case None => text(Pretty.prettyClass(state)) +: state.t.view.map(typ =>
              text(typ.canonicalPType.toString)
            )
          case Some(nested) => text(Pretty.prettyClass(state)) +: state.t.map(typ =>
              text(typ.canonicalPType.toString)
            ) :+ prettyAggStateSignatures(nested)
        })
    }
  }

  def prettyPhysicalAggSigs(aggSigs: Seq[PhysicalAggSig]): Doc =
    list(aggSigs.view.map(prettyPhysicalAggSig))

  def prettyPhysicalAggSig(aggSig: PhysicalAggSig): Doc = {
    aggSig match {
      case GroupedAggSig(t, nested) =>
        fillList("Grouped", t.canonicalPType.toString, prettyPhysicalAggSigs(nested))
      case ArrayLenAggSig(kl, nested) =>
        fillList("ArrayLen", Pretty.prettyBooleanLiteral(kl), prettyPhysicalAggSigs(nested))
      case AggElementsAggSig(nested) =>
        fillList("AggElements", prettyPhysicalAggSigs(nested))
      case PhysicalAggSig(op, state) =>
        fillList(Pretty.prettyClass(op), prettyAggStateSignature(state))
    }
  }

  def single(d: Doc): Iterable[Doc] = RichIterable.single(d)

  /* Dispatch is on the node's tag, in two levels, as in TypeCheck: `header` switches on the family,
   * and each family's method switches on the tag. One match over every node class was over 8000
   * bytes of bytecode, so HotSpot never compiled it. */
  def header(ir: BaseIR, elideBindings: Boolean = false): Iterable[Doc] =
    (ir.tag.family.toInt: @switch) match {
      case Family.Const => headerConst(ir)
      case Family.Core => headerCore(ir, elideBindings)
      case Family.Struct => headerStruct(ir)
      case Family.Stream => headerStream(ir, elideBindings)
      case Family.Collection => headerCollection(ir, elideBindings)
      case Family.Agg => headerAgg(ir, elideBindings)
      case Family.IO => headerIO(ir)
      case Family.NDArray => headerNDArray(ir, elideBindings)
      case Family.Random => headerRandom(ir)
      case Family.Relational => headerRelational(ir, elideBindings)
      case Family.Table => headerTable(ir)
      case Family.Matrix => headerMatrix(ir)
      case Family.BlockMatrix => headerBlockMatrix(ir, elideBindings)
      case _ => Iterable.empty
    }

  private def headerConst(ir: BaseIR): Iterable[Doc] =
    (ir.tag.toInt: @switch) match {
      case Tag.I32 =>
        single(ir.asInstanceOf[I32].x.toString)
      case Tag.I64 =>
        single(ir.asInstanceOf[I64].x.toString)
      case Tag.F32 =>
        single(ir.asInstanceOf[F32].x.toString)
      case Tag.F64 =>
        single(ir.asInstanceOf[F64].x.toString)
      case Tag.Str =>
        val x = ir.asInstanceOf[Str].x
        single(prettyStringLiteral(if (elideLiterals && x.length > 13) x.take(10) + "..." else x))
      case Tag.NA =>
        single(ir.asInstanceOf[NA]._typ.parsableString())
      case Tag.Literal =>
        val x = ir.asInstanceOf[Literal]
        FastSeq(
          x._typ.parsableString(),
          if (!elideLiterals)
            prettyStringLiteral(
              JsonMethods.compact(JSONAnnotationImpex.exportAnnotation(x.value, x._typ))
            )
          else
            "<literal value>",
        )
      case Tag.EncodedLiteral =>
        single(ir.asInstanceOf[EncodedLiteral].codec.encodedVirtualType.parsableString())
      case _ =>
        Iterable.empty
    }

  private def headerCore(ir: BaseIR, elideBindings: Boolean): Iterable[Doc] =
    (ir.tag.toInt: @switch) match {
      case Tag.Cast =>
        single(ir.asInstanceOf[Cast]._typ.parsableString())
      case Tag.UUID4 =>
        single(prettyIdentifier(ir.asInstanceOf[UUID4].id))
      case Tag.Block =>
        if (elideBindings) Iterable.empty
        else
          ir.asInstanceOf[Block].bindings.flatMap { b =>
            val bindType = b.scope match {
              case Scope.EVAL => "eval"
              case Scope.AGG => "agg"
              case Scope.SCAN => "scan"
            }
            FastSeq(text(bindType), prettyName(b.name))
          }
      case Tag.TailLoop =>
        if (elideBindings) Iterable.empty
        else {
          val x = ir.asInstanceOf[TailLoop]
          FastSeq(
            prettyName(x.name),
            prettyNames(x.params.map(_._1)),
            x.resultType.parsableString(),
          )
        }
      case Tag.Recur =>
        if (elideBindings) Iterable.empty
        else FastSeq(prettyName(ir.asInstanceOf[Recur].name))
      // For debug purposes, Ref can also print its type:
      // FastSeq(prettyIdentifier(name), t.parsableString())
      case Tag.Ref =>
        single(prettyName(ir.asInstanceOf[Ref].name))
      case Tag.ApplyBinaryPrimOp =>
        single(Pretty.prettyClass(ir.asInstanceOf[ApplyBinaryPrimOp].op))
      case Tag.ApplyUnaryPrimOp =>
        single(Pretty.prettyClass(ir.asInstanceOf[ApplyUnaryPrimOp].op))
      case Tag.ApplyComparisonOp =>
        single(ir.asInstanceOf[ApplyComparisonOp].op.render())
      case Tag.ApplyIR =>
        val x = ir.asInstanceOf[ApplyIR]
        FastSeq(
          s"${x.errorID}",
          prettyIdentifier(x.function),
          prettyTypes(x.typeArgs),
          x.typ.parsableString(),
        )
      case Tag.Apply =>
        val x = ir.asInstanceOf[Apply]
        FastSeq(
          s"${x.errorID}",
          prettyIdentifier(x.function),
          prettyTypes(x.typeArgs),
          x.returnType.parsableString(),
        )
      case Tag.ApplySpecial =>
        val x = ir.asInstanceOf[ApplySpecial]
        FastSeq(
          s"${x.errorID}",
          prettyIdentifier(x.function),
          prettyTypes(x.typeArgs),
          x.returnType.parsableString(),
        )
      case Tag.In =>
        val x = ir.asInstanceOf[In]
        FastSeq(x._typ.toString, x.i.toString)
      case Tag.Die =>
        val x = ir.asInstanceOf[Die]
        FastSeq(x._typ.parsableString(), x.errorID.toString)
      case _ =>
        Iterable.empty
    }

  private def headerStruct(ir: BaseIR): Iterable[Doc] =
    (ir.tag.toInt: @switch) match {
      case Tag.CastRename =>
        single(ir.asInstanceOf[CastRename]._typ.parsableString())
      case Tag.GetField =>
        single(prettyIdentifier(ir.asInstanceOf[GetField].name))
      case Tag.GetTupleElement =>
        single(ir.asInstanceOf[GetTupleElement].idx.toString)
      case Tag.MakeTuple =>
        FastSeq(prettyInts(ir.asInstanceOf[MakeTuple].fields.map(_._1), elideLiterals))
      case Tag.SelectFields =>
        single(fillList(ir.asInstanceOf[SelectFields].fields.view.map(f =>
          text(prettyIdentifier(f))
        )))
      case _ =>
        Iterable.empty
    }

  private def headerStream(ir: BaseIR, elideBindings: Boolean): Iterable[Doc] =
    (ir.tag.toInt: @switch) match {
      case Tag.MakeStream =>
        val x = ir.asInstanceOf[MakeStream]
        FastSeq(
          x._typ.parsableString(),
          Pretty.prettyBooleanLiteral(x.requiresMemoryManagementPerElement),
        )
      case Tag.StreamBufferedAggregate =>
        val x = ir.asInstanceOf[StreamBufferedAggregate]
        ArraySeq(prettyName(x.name), prettyPhysicalAggSigs(x.aggSignature), prettyInt(x.bufferSize))
      case Tag.StreamIota =>
        FastSeq(Pretty.prettyBooleanLiteral(
          ir.asInstanceOf[StreamIota].requiresMemoryManagementPerElement
        ))
      case Tag.StreamRange =>
        val x = ir.asInstanceOf[StreamRange]
        FastSeq(
          x.errorID.toString,
          Pretty.prettyBooleanLiteral(x.requiresMemoryManagementPerElement),
        )
      case Tag.ToStream =>
        single(Pretty.prettyBooleanLiteral(
          ir.asInstanceOf[ToStream].requiresMemoryManagementPerElement
        ))
      case Tag.StreamMap =>
        if (elideBindings) Iterable.empty
        else single(prettyName(ir.asInstanceOf[StreamMap].name))
      case Tag.StreamZip =>
        val x = ir.asInstanceOf[StreamZip]
        val behavior = x.behavior match {
          case ArrayZipBehavior.AssertSameLength => "AssertSameLength"
          case ArrayZipBehavior.TakeMinLength => "TakeMinLength"
          case ArrayZipBehavior.ExtendNA => "ExtendNA"
          case ArrayZipBehavior.AssumeSameLength => "AssumeSameLength"
        }
        if (elideBindings)
          FastSeq(x.errorID.toString, behavior)
        else
          FastSeq(x.errorID.toString, behavior, prettyNames(x.names))
      case Tag.StreamZipJoin =>
        if (elideBindings) Iterable.empty
        else {
          val x = ir.asInstanceOf[StreamZipJoin]
          FastSeq(
            x.as.length.toString,
            prettyIdentifiers(x.key),
            prettyName(x.curKey),
            prettyName(x.curVals),
          )
        }
      case Tag.StreamZipJoinProducers =>
        if (elideBindings) Iterable.empty
        else {
          val x = ir.asInstanceOf[StreamZipJoinProducers]
          FastSeq(
            prettyIdentifiers(x.key),
            prettyName(x.ctxName),
            prettyName(x.curKey),
            prettyName(x.curVals),
          )
        }
      case Tag.StreamMultiMerge =>
        single(prettyIdentifiers(ir.asInstanceOf[StreamMultiMerge].key))
      case Tag.StreamFilter =>
        if (elideBindings) Iterable.empty
        else single(prettyName(ir.asInstanceOf[StreamFilter].name))
      case Tag.StreamTakeWhile =>
        if (elideBindings) Iterable.empty
        else single(prettyName(ir.asInstanceOf[StreamTakeWhile].elementName))
      case Tag.StreamDropWhile =>
        if (elideBindings) Iterable.empty
        else single(prettyName(ir.asInstanceOf[StreamDropWhile].elementName))
      case Tag.StreamFlatMap =>
        if (elideBindings) Iterable.empty
        else single(prettyName(ir.asInstanceOf[StreamFlatMap].name))
      case Tag.StreamFold =>
        if (elideBindings) Iterable.empty
        else {
          val x = ir.asInstanceOf[StreamFold]
          FastSeq(prettyName(x.accumName), prettyName(x.valueName))
        }
      case Tag.StreamFold2 =>
        if (elideBindings) Iterable.empty
        else {
          val x = ir.asInstanceOf[StreamFold2]
          FastSeq(prettyNames(x.accum.map(_._1)), prettyName(x.valueName))
        }
      case Tag.StreamScan =>
        if (elideBindings) Iterable.empty
        else {
          val x = ir.asInstanceOf[StreamScan]
          FastSeq(prettyName(x.accumName), prettyName(x.valueName))
        }
      case Tag.StreamWhiten =>
        val x = ir.asInstanceOf[StreamWhiten]
        FastSeq(
          prettyIdentifier(x.newChunk),
          prettyIdentifier(x.prevWindow),
          x.vecSize.toString,
          x.windowSize.toString,
          x.chunkSize.toString,
          x.blockSize.toString,
          Pretty.prettyBooleanLiteral(x.normalizeAfterWhiten),
        )
      case Tag.StreamJoinRightDistinct =>
        val x = ir.asInstanceOf[StreamJoinRightDistinct]
        if (elideBindings)
          FastSeq(prettyIdentifiers(x.lKey), prettyIdentifiers(x.rKey), x.joinType)
        else
          FastSeq(
            prettyIdentifiers(x.lKey),
            prettyIdentifiers(x.rKey),
            prettyName(x.l),
            prettyName(x.r),
            x.joinType,
          )
      case Tag.StreamLeftIntervalJoin =>
        val x = ir.asInstanceOf[StreamLeftIntervalJoin]
        val builder = ArraySeq.newBuilder[Doc]
        builder.sizeHint(if (elideBindings) 2 else 4)

        builder += prettyIdentifier(x.lKeyFieldName)
        builder += prettyIdentifier(x.rIntervalFieldName)

        if (!elideBindings) {
          builder += prettyName(x.lname)
          builder += prettyName(x.rname)
        }

        builder.result()
      case Tag.StreamFor =>
        if (elideBindings) Iterable.empty
        else single(prettyName(ir.asInstanceOf[StreamFor].valueName))
      case Tag.StreamAgg =>
        if (elideBindings) Iterable.empty
        else single(prettyName(ir.asInstanceOf[StreamAgg].name))
      case Tag.StreamAggScan =>
        if (elideBindings) Iterable.empty
        else single(prettyName(ir.asInstanceOf[StreamAggScan].name))
      case Tag.StreamGroupByKey =>
        val x = ir.asInstanceOf[StreamGroupByKey]
        FastSeq(prettyIdentifiers(x.key), prettyBooleanLiteral(x.missingEqual))
      case _ =>
        Iterable.empty
    }

  private def headerCollection(ir: BaseIR, elideBindings: Boolean): Iterable[Doc] =
    (ir.tag.toInt: @switch) match {
      case Tag.MakeArray =>
        single(ir.asInstanceOf[MakeArray]._typ.parsableString())
      case Tag.ArraySort =>
        if (elideBindings) Iterable.empty
        else {
          val x = ir.asInstanceOf[ArraySort]
          FastSeq(prettyName(x.left), prettyName(x.right))
        }
      case Tag.ArrayRef =>
        single(s"${ir.asInstanceOf[ArrayRef].errorID}")
      case Tag.LowerBoundOnOrderedCollection =>
        single(Pretty.prettyBooleanLiteral(ir.asInstanceOf[LowerBoundOnOrderedCollection].onKey))
      case Tag.CollectDistributedArray =>
        if (elideBindings) Iterable.empty
        else {
          val x = ir.asInstanceOf[CollectDistributedArray]
          FastSeq(x.staticID, prettyName(x.cname), prettyName(x.gname))
        }
      case _ =>
        Iterable.empty
    }

  private def headerAgg(ir: BaseIR, elideBindings: Boolean): Iterable[Doc] =
    (ir.tag.toInt: @switch) match {
      case Tag.ApplyAggOp =>
        single(Pretty.prettyClass(ir.asInstanceOf[ApplyAggOp].op))
      case Tag.ApplyScanOp =>
        single(Pretty.prettyClass(ir.asInstanceOf[ApplyScanOp].op))
      case Tag.InitOp =>
        val x = ir.asInstanceOf[InitOp]
        FastSeq(x.i.toString, prettyPhysicalAggSig(x.aggSig))
      case Tag.SeqOp =>
        val x = ir.asInstanceOf[SeqOp]
        FastSeq(x.i.toString, prettyPhysicalAggSig(x.aggSig))
      case Tag.CombOp =>
        val x = ir.asInstanceOf[CombOp]
        FastSeq(x.i1.toString, x.i2.toString, prettyPhysicalAggSig(x.aggSig))
      case Tag.ResultOp =>
        val x = ir.asInstanceOf[ResultOp]
        FastSeq(x.idx.toString, prettyPhysicalAggSig(x.aggSig))
      case Tag.AggStateValue =>
        val x = ir.asInstanceOf[AggStateValue]
        FastSeq(x.i.toString, prettyAggStateSignature(x.aggSig))
      case Tag.InitFromSerializedValue =>
        val x = ir.asInstanceOf[InitFromSerializedValue]
        FastSeq(x.i.toString, prettyAggStateSignature(x.aggSig))
      case Tag.CombOpValue =>
        val x = ir.asInstanceOf[CombOpValue]
        FastSeq(x.i.toString, prettyPhysicalAggSig(x.aggSig))
      case Tag.SerializeAggs =>
        val x = ir.asInstanceOf[SerializeAggs]
        FastSeq(
          x.startIdx.toString,
          x.serializedIdx.toString,
          prettyStringLiteral(x.spec.toString),
          prettyAggStateSignatures(x.aggSigs),
        )
      case Tag.DeserializeAggs =>
        val x = ir.asInstanceOf[DeserializeAggs]
        FastSeq(
          x.startIdx.toString,
          x.serializedIdx.toString,
          prettyStringLiteral(x.spec.toString),
          prettyAggStateSignatures(x.aggSigs),
        )
      case Tag.RunAgg =>
        single(prettyAggStateSignatures(ir.asInstanceOf[RunAgg].signature))
      case Tag.RunAggScan =>
        val x = ir.asInstanceOf[RunAggScan]
        FastSeq(prettyName(x.name), prettyAggStateSignatures(x.signature))
      case Tag.AggFold =>
        val x = ir.asInstanceOf[AggFold]
        if (elideBindings)
          single(Pretty.prettyBooleanLiteral(x.isScan))
        else
          FastSeq(
            prettyName(x.accumName),
            prettyName(x.otherAccumName),
            Pretty.prettyBooleanLiteral(x.isScan),
          )
      case Tag.AggExplode =>
        val x = ir.asInstanceOf[AggExplode]
        if (elideBindings) single(Pretty.prettyBooleanLiteral(x.isScan))
        else FastSeq(prettyName(x.name), Pretty.prettyBooleanLiteral(x.isScan))
      case Tag.AggFilter =>
        single(Pretty.prettyBooleanLiteral(ir.asInstanceOf[AggFilter].isScan))
      case Tag.AggGroupBy =>
        single(Pretty.prettyBooleanLiteral(ir.asInstanceOf[AggGroupBy].isScan))
      case Tag.AggArrayPerElement =>
        val x = ir.asInstanceOf[AggArrayPerElement]
        if (elideBindings)
          FastSeq(
            Pretty.prettyBooleanLiteral(x.isScan),
            Pretty.prettyBooleanLiteral(x.knownLength.isDefined),
          )
        else
          FastSeq(
            prettyName(x.elementName),
            prettyName(x.indexName),
            Pretty.prettyBooleanLiteral(x.isScan),
            Pretty.prettyBooleanLiteral(x.knownLength.isDefined),
          )
      case _ =>
        Iterable.empty
    }

  private def headerIO(ir: BaseIR): Iterable[Doc] =
    (ir.tag.toInt: @switch) match {
      case Tag.ReadPartition =>
        val x = ir.asInstanceOf[ReadPartition]
        FastSeq(
          x.rowType.parsableString(),
          prettyStringLiteral(JsonMethods.compact(x.reader.toJValue)),
        )
      case Tag.WritePartition =>
        single(prettyStringLiteral(JsonMethods.compact(
          ir.asInstanceOf[WritePartition].writer.toJValue
        )))
      case Tag.WriteMetadata =>
        single(prettyStringLiteral(
          JsonMethods.compact(ir.asInstanceOf[WriteMetadata].writer.toJValue),
          elide = elideLiterals,
        ))
      case Tag.ReadValue =>
        val x = ir.asInstanceOf[ReadValue]
        FastSeq(
          prettyStringLiteral(JsonMethods.compact(x.reader.toJValue)),
          x.requestedType.parsableString(),
        )
      case Tag.WriteValue =>
        single(prettyStringLiteral(JsonMethods.compact(
          ir.asInstanceOf[WriteValue].writer.toJValue
        )))
      case _ =>
        Iterable.empty
    }

  private def headerNDArray(ir: BaseIR, elideBindings: Boolean): Iterable[Doc] =
    (ir.tag.toInt: @switch) match {
      case Tag.NDArrayMap =>
        if (elideBindings) Iterable.empty
        else single(prettyName(ir.asInstanceOf[NDArrayMap].valueName))
      case Tag.NDArrayMap2 =>
        val x = ir.asInstanceOf[NDArrayMap2]
        if (elideBindings) single(s"${x.errorID}")
        else FastSeq(s"${x.errorID}", prettyName(x.lName), prettyName(x.rName))
      case Tag.NDArrayReindex =>
        single(prettyInts(ir.asInstanceOf[NDArrayReindex].indexExpr, elideLiterals))
      case Tag.NDArrayConcat =>
        single(ir.asInstanceOf[NDArrayConcat].axis.toString)
      case Tag.NDArrayAgg =>
        single(prettyInts(ir.asInstanceOf[NDArrayAgg].axes, elideLiterals))
      case Tag.NDArrayRef =>
        single(s"${ir.asInstanceOf[NDArrayRef].errorID}")
      case Tag.NDArrayReshape =>
        single(s"${ir.asInstanceOf[NDArrayReshape].errorID}")
      case Tag.NDArrayMatMul =>
        single(s"${ir.asInstanceOf[NDArrayMatMul].errorID}")
      case Tag.NDArrayQR =>
        val x = ir.asInstanceOf[NDArrayQR]
        FastSeq(x.errorID.toString, x.mode)
      case Tag.NDArraySVD =>
        val x = ir.asInstanceOf[NDArraySVD]
        FastSeq(x.errorID.toString, x.fullMatrices.toString, x.computeUV.toString)
      case Tag.NDArrayEigh =>
        val x = ir.asInstanceOf[NDArrayEigh]
        FastSeq(x.errorID.toString, x.eigvalsOnly.toString)
      case Tag.NDArrayInv =>
        single(s"${ir.asInstanceOf[NDArrayInv].errorID}")
      case Tag.MakeNDArray =>
        FastSeq(ir.asInstanceOf[MakeNDArray].errorID.toString)
      case _ =>
        Iterable.empty
    }

  private def headerRandom(ir: BaseIR): Iterable[Doc] =
    (ir.tag.toInt: @switch) match {
      case Tag.RNGSplitStatic => single(prettyLong(ir.asInstanceOf[RNGSplitStatic].staticUid))
      case _ => Iterable.empty
    }

  private def headerRelational(ir: BaseIR, elideBindings: Boolean): Iterable[Doc] =
    (ir.tag.toInt: @switch) match {
      case Tag.RelationalRef =>
        val x = ir.asInstanceOf[RelationalRef]
        if (elideBindings) single(x._typ.parsableString())
        else FastSeq(prettyName(x.name), x._typ.parsableString())
      case Tag.RelationalLet =>
        if (elideBindings) Iterable.empty
        else single(prettyName(ir.asInstanceOf[RelationalLet].name))
      case _ =>
        Iterable.empty
    }

  private def headerTable(ir: BaseIR): Iterable[Doc] =
    (ir.tag.toInt: @switch) match {
      case Tag.TableRead =>
        val x = ir.asInstanceOf[TableRead]
        FastSeq(
          if (x.typ == x.tr.fullType) "None" else x.typ.parsableString(),
          Pretty.prettyBooleanLiteral(x.dropRows),
          if (elideLiterals) x.tr.renderShort()
          else s""""${StringEscapeUtils.escapeString(JsonMethods.compact(x.tr.toJValue))}"""",
        )
      case Tag.TableWrite =>
        val writer = ir.asInstanceOf[TableWrite].writer
        single(
          s""""${StringEscapeUtils.escapeString(Serialization.write(writer)(TableWriter.formats))}""""
        )
      case Tag.TableMultiWrite =>
        val writer = ir.asInstanceOf[TableMultiWrite].writer
        single(
          s""""${StringEscapeUtils.escapeString(Serialization.write(writer)(WrappedMatrixNativeMultiWriter.formats))}""""
        )
      case Tag.TableKeyBy =>
        val x = ir.asInstanceOf[TableKeyBy]
        FastSeq(
          prettyIdentifiers(x.keys),
          Pretty.prettyBooleanLiteral(x.isSorted),
          prettyIntOpt(x.nPartitions),
        )
      case Tag.TableRange =>
        val x = ir.asInstanceOf[TableRange]
        FastSeq(x.n.toString, x.nPartitions.toString)
      case Tag.TableRepartition =>
        val x = ir.asInstanceOf[TableRepartition]
        FastSeq(x.n.toString, x.strategy.toString)
      case Tag.TableHead =>
        single(ir.asInstanceOf[TableHead].n.toString)
      case Tag.TableTail =>
        single(ir.asInstanceOf[TableTail].n.toString)
      case Tag.TableJoin =>
        val x = ir.asInstanceOf[TableJoin]
        FastSeq(x.joinType, x.joinKey.toString)
      case Tag.TableLeftJoinRightDistinct =>
        single(prettyIdentifier(ir.asInstanceOf[TableLeftJoinRightDistinct].root))
      case Tag.TableIntervalJoin =>
        val x = ir.asInstanceOf[TableIntervalJoin]
        FastSeq(prettyIdentifier(x.root), Pretty.prettyBooleanLiteral(x.product))
      case Tag.TableMultiWayZipJoin =>
        val x = ir.asInstanceOf[TableMultiWayZipJoin]
        FastSeq(prettyStringLiteral(x.fieldName), prettyStringLiteral(x.globalName))
      case Tag.TableKeyByAndAggregate =>
        val x = ir.asInstanceOf[TableKeyByAndAggregate]
        FastSeq(prettyIntOpt(x.nPartitions), x.bufferSize.toString)
      case Tag.TableExplode =>
        single(prettyStrings(ir.asInstanceOf[TableExplode].path))
      case Tag.TableMapPartitions =>
        val x = ir.asInstanceOf[TableMapPartitions]
        FastSeq(
          prettyName(x.globalName),
          prettyName(x.partitionStreamName),
          x.requestedKey.toString,
          x.allowedOverlap.toString,
        )
      case Tag.TableParallelize =>
        single(prettyIntOpt(ir.asInstanceOf[TableParallelize].nPartitions))
      case Tag.TableOrderBy =>
        single(prettySortFields(ir.asInstanceOf[TableOrderBy].sortFields))
      case Tag.TableToTableApply =>
        val function = ir.asInstanceOf[TableToTableApply].function
        single(prettyStringLiteral(JsonMethods.compact(function.toJValue)))
      case Tag.TableToValueApply =>
        val function = ir.asInstanceOf[TableToValueApply].function
        single(prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats)))
      case Tag.BlockMatrixToTableApply =>
        val function = ir.asInstanceOf[BlockMatrixToTableApply].function
        single(prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats)))
      case Tag.TableGen =>
        val x = ir.asInstanceOf[TableGen]
        implicit val jsonFormats = DefaultFormats
        FastSeq(
          prettyName(x.cname),
          prettyName(x.gname), {
            val boundsJson =
              Serialization.write(x.partitioner.rangeBounds.map(_.toJSON(
                x.partitioner.kType.export
              )))
            list(
              "Partitioner " + x.partitioner.kType.parsableString() + prettyStringLiteral(
                boundsJson
              )
            )
          },
          text(x.errorId.toString),
        )
      case Tag.TableRename =>
        val x = ir.asInstanceOf[TableRename]
        FastSeq(
          prettyStrings(ArraySeq.from(x.rowMap.keys)),
          prettyStrings(ArraySeq.from(x.rowMap.values)),
          prettyStrings(ArraySeq.from(x.globalMap.keys)),
          prettyStrings(ArraySeq.from(x.globalMap.values)),
        )
      case Tag.TableFilterIntervals =>
        val x = ir.asInstanceOf[TableFilterIntervals]
        FastSeq(
          x.child.typ.keyType.parsableString(),
          prettyStringLiteral(Serialization.write(
            JSONAnnotationImpex.exportAnnotation(
              x.intervals,
              TArray(TInterval(x.child.typ.keyType)),
            )
          )(RelationalSpec.formats)),
          Pretty.prettyBooleanLiteral(x.keep),
        )
      case Tag.RelationalLetTable =>
        single(prettyName(ir.asInstanceOf[RelationalLetTable].name))
      case _ =>
        Iterable.empty
    }

  private def headerMatrix(ir: BaseIR): Iterable[Doc] =
    (ir.tag.toInt: @switch) match {
      case Tag.MatrixRead =>
        val x = ir.asInstanceOf[MatrixRead]
        FastSeq(
          if (x.typ == x.reader.fullMatrixType) "None" else x.typ.parsableString(),
          Pretty.prettyBooleanLiteral(x.dropCols),
          Pretty.prettyBooleanLiteral(x.dropRows),
          if (elideLiterals) x.reader.renderShort()
          else s""""${StringEscapeUtils.escapeString(JsonMethods.compact(x.reader.toJValue))}"""",
        )
      case Tag.MatrixWrite =>
        val writer = ir.asInstanceOf[MatrixWrite].writer
        single(
          s""""${StringEscapeUtils.escapeString(Serialization.write(writer)(MatrixWriter.formats))}""""
        )
      case Tag.MatrixMultiWrite =>
        val writer = ir.asInstanceOf[MatrixMultiWrite].writer
        single(
          s""""${StringEscapeUtils.escapeString(Serialization.write(writer)(MatrixNativeMultiWriter.formats))}""""
        )
      case Tag.MatrixRowsHead =>
        single(ir.asInstanceOf[MatrixRowsHead].n.toString)
      case Tag.MatrixColsHead =>
        single(ir.asInstanceOf[MatrixColsHead].n.toString)
      case Tag.MatrixRowsTail =>
        single(ir.asInstanceOf[MatrixRowsTail].n.toString)
      case Tag.MatrixColsTail =>
        single(ir.asInstanceOf[MatrixColsTail].n.toString)
      case Tag.MatrixAnnotateRowsTable =>
        val x = ir.asInstanceOf[MatrixAnnotateRowsTable]
        FastSeq(prettyStringLiteral(x.root), Pretty.prettyBooleanLiteral(x.product))
      case Tag.MatrixAnnotateColsTable =>
        single(prettyStringLiteral(ir.asInstanceOf[MatrixAnnotateColsTable].root))
      case Tag.MatrixExplodeRows =>
        single(prettyIdentifiers(ir.asInstanceOf[MatrixExplodeRows].path))
      case Tag.MatrixExplodeCols =>
        single(prettyIdentifiers(ir.asInstanceOf[MatrixExplodeCols].path))
      case Tag.MatrixRepartition =>
        val x = ir.asInstanceOf[MatrixRepartition]
        single(s"${x.n} ${x.strategy}")
      case Tag.MatrixChooseCols =>
        single(prettyInts(ir.asInstanceOf[MatrixChooseCols].oldIndices, elideLiterals))
      case Tag.MatrixMapCols =>
        single(prettyStringsOpt(ir.asInstanceOf[MatrixMapCols].newKey))
      case Tag.MatrixUnionCols =>
        single(ir.asInstanceOf[MatrixUnionCols].joinType)
      case Tag.MatrixKeyRowsBy =>
        val x = ir.asInstanceOf[MatrixKeyRowsBy]
        FastSeq(prettyIdentifiers(x.keys), Pretty.prettyBooleanLiteral(x.isSorted))
      case Tag.CastMatrixToTable =>
        val x = ir.asInstanceOf[CastMatrixToTable]
        FastSeq(prettyStringLiteral(x.entriesFieldName), prettyStringLiteral(x.colsFieldName))
      case Tag.CastTableToMatrix =>
        val x = ir.asInstanceOf[CastTableToMatrix]
        FastSeq(
          prettyIdentifier(x.entriesFieldName),
          prettyIdentifier(x.colsFieldName),
          prettyIdentifiers(x.colKey),
        )
      case Tag.MatrixToMatrixApply =>
        val function = ir.asInstanceOf[MatrixToMatrixApply].function
        single(prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats)))
      case Tag.MatrixToTableApply =>
        val function = ir.asInstanceOf[MatrixToTableApply].function
        single(prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats)))
      case Tag.MatrixToValueApply =>
        val function = ir.asInstanceOf[MatrixToValueApply].function
        single(prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats)))
      case Tag.MatrixRename =>
        val x = ir.asInstanceOf[MatrixRename]
        FastSeq(
          prettyStrings(ArraySeq.from(x.globalMap.keys)),
          prettyStrings(ArraySeq.from(x.globalMap.values)),
          prettyStrings(ArraySeq.from(x.colMap.keys)),
          prettyStrings(ArraySeq.from(x.colMap.values)),
          prettyStrings(ArraySeq.from(x.rowMap.keys)),
          prettyStrings(ArraySeq.from(x.rowMap.values)),
          prettyStrings(ArraySeq.from(x.entryMap.keys)),
          prettyStrings(ArraySeq.from(x.entryMap.values)),
        )
      case Tag.MatrixFilterIntervals =>
        val x = ir.asInstanceOf[MatrixFilterIntervals]
        FastSeq(
          x.child.typ.rowType.parsableString(),
          prettyStringLiteral(Serialization.write(
            JSONAnnotationImpex.exportAnnotation(
              x.intervals,
              TArray(TInterval(x.child.typ.rowKeyStruct)),
            )
          )(RelationalSpec.formats)),
          Pretty.prettyBooleanLiteral(x.keep),
        )
      case Tag.RelationalLetMatrixTable =>
        single(prettyName(ir.asInstanceOf[RelationalLetMatrixTable].name))
      case _ =>
        Iterable.empty
    }

  private def headerBlockMatrix(ir: BaseIR, elideBindings: Boolean): Iterable[Doc] =
    (ir.tag.toInt: @switch) match {
      case Tag.BlockMatrixRead =>
        val reader = ir.asInstanceOf[BlockMatrixRead].reader
        single(s""""${StringEscapeUtils.escapeString(JsonMethods.compact(reader.toJValue))}"""")
      case Tag.BlockMatrixWrite =>
        val writer = ir.asInstanceOf[BlockMatrixWrite].writer
        single(
          s""""${StringEscapeUtils.escapeString(Serialization.write(writer)(BlockMatrixWriter.formats))}""""
        )
      case Tag.BlockMatrixMultiWrite =>
        val writer = ir.asInstanceOf[BlockMatrixMultiWrite].writer
        single(
          s""""${StringEscapeUtils.escapeString(Serialization.write(writer)(BlockMatrixWriter.formats))}""""
        )
      case Tag.BlockMatrixBroadcast =>
        val x = ir.asInstanceOf[BlockMatrixBroadcast]
        FastSeq(
          prettyInts(x.inIndexExpr, elideLiterals),
          prettyLongs(x.shape, elideLiterals),
          x.blockSize.toString,
        )
      case Tag.BlockMatrixAgg =>
        single(prettyInts(ir.asInstanceOf[BlockMatrixAgg].axesToSumOut, elideLiterals))
      case Tag.BlockMatrixSlice =>
        single(fillList(ir.asInstanceOf[BlockMatrixSlice].slices.view.map(slice =>
          prettyLongs(slice, elideLiterals)
        )))
      case Tag.ValueToBlockMatrix =>
        val x = ir.asInstanceOf[ValueToBlockMatrix]
        FastSeq(prettyLongs(x.shape, elideLiterals), x.blockSize.toString)
      case Tag.BlockMatrixFilter =>
        single(fillList(ir.asInstanceOf[BlockMatrixFilter].indices.view.map(indices =>
          prettyLongs(indices, elideLiterals)
        )))
      case Tag.BlockMatrixSparsify =>
        single(ir.asInstanceOf[BlockMatrixSparsify].sparsifier.pretty())
      case Tag.BlockMatrixRandom =>
        val x = ir.asInstanceOf[BlockMatrixRandom]
        FastSeq(
          x.staticUID.toString,
          Pretty.prettyBooleanLiteral(x.gaussian),
          prettyLongs(x.shape, elideLiterals),
          x.blockSize.toString,
        )
      case Tag.BlockMatrixMap =>
        val x = ir.asInstanceOf[BlockMatrixMap]
        if (elideBindings) single(Pretty.prettyBooleanLiteral(x.needsDense))
        else FastSeq(prettyName(x.eltName), Pretty.prettyBooleanLiteral(x.needsDense))
      case Tag.BlockMatrixMap2 =>
        val x = ir.asInstanceOf[BlockMatrixMap2]
        if (elideBindings)
          single(Pretty.prettyClass(x.sparsityStrategy))
        else
          FastSeq(
            prettyName(x.leftName),
            prettyName(x.rightName),
            Pretty.prettyClass(x.sparsityStrategy),
          )
      case Tag.BlockMatrixToValueApply =>
        val function = ir.asInstanceOf[BlockMatrixToValueApply].function
        single(prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats)))
      case _ =>
        Iterable.empty
    }

  def apply(ir: BaseIR): String = if (useSSA)
    ssaStyle(ir)
  else
    sexprStyle(ir)

  def sexprStyle(ir: BaseIR): String = {
    def prettySeq(xs: Seq[BaseIR]): Doc =
      list(xs.view.map(pretty))

    def pretty(ir: BaseIR): Doc = {

      val body: Iterable[Doc] = ir match {
        case MakeStruct(fields) =>
          fields.view.map { case (n, a) =>
            list(n, pretty(a))
          }
        case ApplyAggOp(initOpArgs, seqOpArgs, _) =>
          FastSeq(prettySeq(initOpArgs), prettySeq(seqOpArgs))
        case ApplyScanOp(initOpArgs, seqOpArgs, _) =>
          FastSeq(prettySeq(initOpArgs), prettySeq(seqOpArgs))
        case InitOp(_, args, _) => single(prettySeq(args))
        case SeqOp(_, args, _) => single(prettySeq(args))
        case InsertFields(old, fields, fieldOrder) =>
          val fieldDocs = fields.view.map { case (n, a) =>
            list(prettyIdentifier(n), pretty(a))
          }
          pretty(old) +: prettyStringsOpt(fieldOrder) +: fieldDocs
        case _ => ir.children.map(pretty)
      }

      /* val pt = ir match{ case ir: IR => if (ir._pType != null) single(ir.pType.toString) case _
       * => Iterable.empty } list(fillSep(text(prettyClass(ir)) +: pt ++ header(ir, elideLiterals))
       * +: body) */
      list(fillSep(text(Pretty.prettyClass(ir)) +: header(ir)) +: body)
    }

    pretty(ir).render(width, ribbonWidth, maxLen)
  }

  def ssaStyle(ir: BaseIR): String = {
    def childIsStrict(ir: BaseIR, i: Int): Boolean = blockArgs(ir, i).isEmpty
    def some(as: (Name, String)*) = Some(as.toFastSeq)

    lazy val matrixBlockArgs =
      some(
        MatrixIR.globalName -> "g",
        MatrixIR.colName -> "col",
        MatrixIR.rowName -> "row",
        MatrixIR.entryName -> "entry",
      )

    lazy val tableBlockArgs =
      some(
        TableIR.globalName -> "g",
        TableIR.rowName -> "row",
      )

    def blockArgs(ir: BaseIR, i: Int): Option[IndexedSeq[(Name, String)]] =
      ir match {
        case AggArrayPerElement(_, elementName, indexName, _, _, _) =>
          if (i == 1) some(elementName -> "elt", indexName -> "idx")
          else None
        case AggExplode(_, name, _, _) =>
          if (i == 1) some(name -> "elt")
          else None
        case _: AggFilter =>
          if (i == 1) some()
          else None
        case AggFold(_, _, _, accumName, otherAccumName, _) =>
          if (i == 1) some(accumName -> "accum")
          else if (i == 2) some(accumName -> "l", otherAccumName -> "r")
          else None
        case _: AggGroupBy =>
          if (i == 1) some()
          else None
        case ArraySort(_, left, right, _) =>
          if (i == 1) some(left -> "l", right -> "r")
          else None
        case BlockMatrixMap(_, eltName, _, _) =>
          if (i == 1) some(eltName -> "elt")
          else None
        case BlockMatrixMap2(_, _, lName, rName, _, _) =>
          if (i == 2) some(lName -> "l", rName -> "r")
          else None
        case _: Coalesce =>
          some()
        case CollectDistributedArray(_, _, cname, gname, _, _, _, _) =>
          if (i == 2) some(cname -> "ctx", gname -> "g")
          else None
        case _: ConsoleLog =>
          if (i == 1) some()
          else None
        case _: If =>
          if (i > 0) some()
          else None
        case _: MatrixAggregate =>
          if (i == 1) matrixBlockArgs
          else None
        case _: MatrixAggregateColsByKey =>
          if (i == 1) matrixBlockArgs
          else if (i == 2) some(MatrixIR.globalName -> "g", MatrixIR.colName -> "col")
          else None
        case _: MatrixAggregateRowsByKey =>
          if (i == 1) matrixBlockArgs
          else if (i == 2) some(MatrixIR.globalName -> "g", MatrixIR.rowName -> "row")
          else None
        case _: MatrixFilterCols =>
          if (i == 1) some(MatrixIR.globalName -> "g", MatrixIR.colName -> "col")
          else None
        case _: MatrixFilterEntries =>
          if (i == 1) matrixBlockArgs
          else None
        case _: MatrixFilterRows =>
          if (i == 1) some(MatrixIR.globalName -> "g", MatrixIR.rowName -> "row")
          else None
        case _: MatrixMapCols =>
          if (i == 1) matrixBlockArgs map { _ :+ (Name("n_rows") -> "n_rows") }
          else None
        case _: MatrixMapEntries =>
          if (i == 1) matrixBlockArgs
          else None
        case _: MatrixMapGlobals =>
          if (i == 1) some(MatrixIR.globalName -> "g")
          else None
        case _: MatrixMapRows =>
          if (i == 1) matrixBlockArgs map { _ :+ (Name("n_cols") -> "n_cols") }
          else None
        case NDArrayMap(_, name, _) =>
          if (i == 1) some(name -> "elt")
          else None
        case NDArrayMap2(_, _, lName, rName, _, _) =>
          if (i == 2) some(lName -> "l_elt", rName -> "r_elt")
          else None
        case RunAggScan(_, name, _, _, _, _) =>
          if (i == 2 || i == 3) some(name -> "elt")
          else None
        case StreamAgg(_, name, _) =>
          if (i == 1) some(name -> "elt")
          else None
        case StreamAggScan(_, name, _) =>
          if (i == 1) some(name -> "elt")
          else None
        case s: StreamBufferedAggregate =>
          if (i >= 2) some(s.name -> "elt")
          else None
        case StreamDropWhile(_, name, _) =>
          if (i == 1) some(name -> "elt")
          else None
        case StreamFlatMap(_, name, _) =>
          if (i == 1) some(name -> "elt")
          else None
        case StreamFilter(_, name, _) =>
          if (i == 1) some(name -> "elt")
          else None
        case StreamFold(_, _, accumName, valueName, _) =>
          if (i == 2) some(accumName -> "accum", valueName -> "elt")
          else None
        case StreamFold2(_, accum, valueName, _, _) =>
          if (i <= accum.length) None
          else if (i < 2 * accum.length + 1) Some((valueName -> "elt") +: accum.map {
            case (name, _) => name -> "accum"
          })
          else Some(accum.map { case (name, _) => name -> "accum" })
        case StreamFor(_, name, _) =>
          if (i == 1) some(name -> "elt")
          else None
        case StreamJoinRightDistinct(_, _, _, _, l, r, _, _) =>
          if (i == 2) some(l -> "l_elt", r -> "r_elt")
          else None
        case StreamLeftIntervalJoin(_, _, _, _, l, r, _) =>
          if (i == 2) some(l -> "l_elt", r -> "r_elts")
          else None
        case StreamMap(_, name, _) =>
          if (i == 1) some(name -> "elt")
          else None
        case StreamScan(_, _, accumName, valueName, _) =>
          if (i == 2) some(accumName -> "accum", valueName -> "elt")
          else None
        case StreamTakeWhile(_, name, _) =>
          if (i == 1) some(name -> "elt")
          else None
        case StreamZip(as, names, _, _, _) =>
          if (i == as.length) Some(names.map(_ -> "elt"))
          else None
        case StreamZipJoin(as, _, curKey, curVals, _) =>
          if (i == as.length) some(curKey -> "key", curVals -> "elts")
          else None
        case StreamZipJoinProducers(_, cname, _, _, k, v, _) =>
          if (i == 1) some(cname -> "elt")
          else if (i == 2) some(k -> "key", v -> "val")
          else None
        case _: Switch =>
          if (i > 0) some()
          else None
        case _: TableAggregate =>
          if (i == 1) tableBlockArgs
          else None
        case _: TableAggregateByKey =>
          if (i == 1) tableBlockArgs
          else None
        case _: TableFilter =>
          if (i == 1) tableBlockArgs
          else None
        case _: TableKeyByAndAggregate =>
          if (i == 1 || i == 2) tableBlockArgs
          else None
        case _: TableMapGlobals =>
          if (i == 1) some(TableIR.globalName -> "g")
          else None
        case TableMapPartitions(_, g, p, _, _, _) =>
          if (i == 1) some(g -> "g", p -> "part")
          else None
        case _: TableMapRows =>
          if (i == 1) tableBlockArgs
          else None
        case TailLoop(name, args, _, _) =>
          if (i == args.length) Some(args.map { case (name, _) =>
            name -> "loopvar"
          } :+ name -> "loop")
          else None
        case _ =>
          None
      }

    var identCounter: Int = 0
    val idents = collection.mutable.Map.empty[String, Int]

    def getIdentBase(ir: BaseIR): String = ir match {
      case True() => "true"
      case False() => "false"
      case I32(i) => s"c$i"
      case stream if stream.typ.isInstanceOf[TStream] => "s"
      case table if table.typ.isInstanceOf[TableType] => "ht"
      case mt if mt.typ.isInstanceOf[MatrixType] => "mt"
      case _ => ""
    }

    def uniqueify(base: String, origName: Option[Name]): String = {
      if (preserveNames && origName.nonEmpty) {
        origName.get.str
      } else {
        if (base.isEmpty) {
          identCounter += 1
          identCounter.toString
        } else if (idents.contains(base)) {
          idents(base) += 1
          if (base.last.isDigit)
            s"${base}_${idents(base)}"
          else
            s"$base${idents(base)}"
        } else {
          idents(base) = 1
          base
        }
      }
    }

    def prettyWithIdent(
      ir: BaseIR,
      bindings: Env[String],
      prefix: String,
      origName: Option[Name],
      scope: Scope = Scope.EVAL,
    ): (Doc, String) = {
      val (pre, body) = pretty(ir, bindings)
      val ident = prefix + uniqueify(getIdentBase(ir), origName)
      val assignmentSymbol = scope match {
        case Scope.EVAL => "="
        case Scope.AGG => "=(agg)"
        case Scope.SCAN => "=(scan)"
      }
      val doc = vsep(pre, hsep(text(ident), assignmentSymbol, body))
      (doc, ident)
    }

    def prettyBlock(ir: BaseIR, newBindings: IndexedSeq[(Name, String)], bindings: Env[String])
      : Doc = {
      val args = newBindings.map { case (name, base) => name -> s"%${uniqueify(base, Some(name))}" }
      val blockBindings = bindings.bindIterable(args)
      val openBlock = if (args.isEmpty)
        text("{")
      else
        concat("{", softline, args.map(_._2).mkString("(", ", ", ") =>"))
      ir match {
        case Ref(name, _) =>
          val body =
            blockBindings.lookupOption(name).getOrElse("#" + uniqueify("undefined_ref", Some(name)))
          concat(openBlock, group(nest(2, concat(line, body, line)), "}"))
        case RelationalRef(name, _) =>
          val body =
            blockBindings.lookupOption(name).getOrElse("#" + uniqueify(
              "undefined_relational_ref",
              Some(name),
            ))
          concat(openBlock, group(nest(2, concat(line, body, line)), "}"))
        case _ =>
          val (pre, body) = pretty(ir, blockBindings)
          concat(openBlock, nest(2, vsep(pre, body)), line, "}")
      }
    }

    def pretty(ir: BaseIR, bindings: Env[String]): (Doc, Doc) = ir match {
      case Block(binds, body) =>
        val (valueDoc, newBindings) =
          binds.foldLeft((empty, bindings)) { case ((valueDoc, bindings), binding) =>
            val (doc, ident) =
              prettyWithIdent(binding.value, bindings, "%", Some(binding.name), binding.scope)
            (concat(valueDoc, doc), bindings.bind(binding.name, ident))
          }
        val (bodyPre, bodyHead) = pretty(body, newBindings)
        (concat(valueDoc, bodyPre), bodyHead)
      case RelationalLet(name, value, body) =>
        val (valueDoc, valueIdent) = prettyWithIdent(value, bindings, "%", Some(name))
        val (bodyPre, bodyHead) = pretty(body, bindings.bind(name, valueIdent))
        (concat(valueDoc, bodyPre), bodyHead)
      case RelationalLetTable(name, value, body) =>
        val (valueDoc, valueIdent) = prettyWithIdent(value, bindings, "%", Some(name))
        val (bodyPre, bodyHead) = pretty(body, bindings.bind(name, valueIdent))
        (concat(valueDoc, bodyPre), bodyHead)
      case RelationalLetMatrixTable(name, value, body) =>
        val (valueDoc, valueIdent) = prettyWithIdent(value, bindings, "%", Some(name))
        val (bodyPre, bodyHead) = pretty(body, bindings.bind(name, valueIdent))
        (concat(valueDoc, bodyPre), bodyHead)
      case _ =>
        val strictChildBodies = mutable.ArrayBuilder.make[Doc]
        val strictChildIdents = for {
          (child, i) <- ir.children.zipWithIndex
          if childIsStrict(ir, i)
        } yield {
          child match {
            case Ref(name, _) =>
              bindings.lookupOption(name).getOrElse("#" + uniqueify("undefined_ref", Some(name)))
            case RelationalRef(name, _) =>
              bindings.lookupOption(name).getOrElse("#" + uniqueify(
                "undefined_relational_ref",
                Some(name),
              ))
            case _ =>
              val (body, ident) = prettyWithIdent(child, bindings, "!", None)
              strictChildBodies += body
              ident
          }
        }

        val nestedBlocks =
          (for {
            (child, i) <- ir.children.zipWithIndex
            if !childIsStrict(ir, i)
          } yield prettyBlock(child, blockArgs(ir, i).get, bindings))

        val attsIterable = header(ir, elideBindings = true)
        val attributes = if (attsIterable.isEmpty) Iterable.empty
        else
          RichIterable.single(concat(attsIterable.intersperse[Doc]("[", ", ", "]")))

        def standardArgs = if (strictChildIdents.isEmpty)
          ""
        else
          strictChildIdents.mkString("(", ", ", ")")

        val head = ir match {
          case MakeStruct(fields) =>
            val args = fields.map(_._1).lazyZip(strictChildIdents).map { (field, value) =>
              s"$field: $value"
            }.mkString("(", ", ", ")")
            hsep(text(Pretty.prettyClass(ir) + args) +: (attributes ++ nestedBlocks))
          case InsertFields(_, fields, _) =>
            val newFields = fields.map(_._1).lazyZip(strictChildIdents.tail).map {
              (field, value) => s"$field: $value"
            }.mkString("(", ", ", ")")
            val args = s" ${strictChildIdents.head} $newFields"
            hsep(text(Pretty.prettyClass(ir) + args) +: (attributes ++ nestedBlocks))
          case ir: If =>
            val Seq(cnsq, altr) = nestedBlocks
            hsep(
              text(s"${Pretty.prettyClass(ir)} ${strictChildIdents.head}"),
              text("then"),
              cnsq,
              text("else"),
              altr,
            )
          case _ =>
            hsep(text(Pretty.prettyClass(ir) + standardArgs) +: (attributes ++ nestedBlocks))
        }

        (hsep(strictChildBodies.result()), head)
    }

    val (pre, head) = pretty(ir, Env.empty)
    vsep(pre, head, empty).render(width, ribbonWidth, maxLen)
  }
}
