package is.hail.expr.ir

import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.agg.{AggElementsAggSig, AggStateSig, ArrayLenAggSig, GroupedAggSig, PhysicalAggSig}
import is.hail.expr.ir.functions.RelationalFunctions
import is.hail.types.virtual.{TArray, TInterval, Type}
import is.hail.utils.{space => _, _}
import is.hail.utils.prettyPrint._
import org.json4s.jackson.{JsonMethods, Serialization}

object Pretty {

  def short(ir: BaseIR, elideLiterals: Boolean = false, maxLen: Int = 100): String = {
    val s = Pretty(ir, elideLiterals = elideLiterals, maxLen = maxLen)
    if (s.length < maxLen) s else s.substring(0, maxLen) + "..."
  }

  def prettyStringLiteral(s: String): String =
    "\"" + StringEscapeUtils.escapeString(s) + "\""

  def fillList(docs: Iterable[Doc], indent: Int = 2): Doc =
    nest(indent, concat(docs.intersperse[Doc](concat("(", softlineAlt), softline, ")")))

  def list(docs: Iterable[Doc], indent: Int = 2): Doc =
    nest(indent, group(concat(docs.intersperse[Doc]("(", line, ")"))))

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

  def prettyBooleanLiteral(b: Boolean): String =
    if (b) "True" else "False"

  def prettyClass(x: AnyRef): String =
    x.getClass.getName.split("\\.").last

  val MAX_VALUES_TO_LOG: Int = 25

  def prettyIntOpt(x: Option[Int]): String =
    x.map(_.toString).getOrElse("None")

  def prettyLongs(x: IndexedSeq[Long], elideLiterals: Boolean): Doc = {
    val truncate = elideLiterals && x.length > MAX_VALUES_TO_LOG
    val view = if (truncate) x.view else x.view(0, MAX_VALUES_TO_LOG)
    val docs = view.map(i => text(i.toString))
    concat(docs.intersperse[Doc](
      concat("(", softlineAlt),
      softline,
      if (truncate) s"... ${ x.length - MAX_VALUES_TO_LOG } more values... )" else ")"))
  }

  def prettyInts(x: IndexedSeq[Int], elideLiterals: Boolean): Doc = {
    val truncate = elideLiterals && x.length > MAX_VALUES_TO_LOG
    val view = if (truncate) x.view else x.view(0, MAX_VALUES_TO_LOG)
    val docs = view.map(i => text(i.toString))
    concat(docs.intersperse[Doc](
      concat("(", softlineAlt),
      softline,
      if (truncate) s"... ${ x.length - MAX_VALUES_TO_LOG } more values... )" else ")"))
  }

  def prettyIdentifiers(x: IndexedSeq[String]): Doc =
    fillList(x.view.map(text))

  def prettyAggStateSignatures(states: Seq[AggStateSig]): Doc =
    list(states.view.map(prettyAggStateSignature))

  def prettyAggStateSignature(state: AggStateSig): Doc = {
    nest(2, concat(
      "(",
      prettyClass(state),
      space,
      fillSep(state.t.view.map(typ => text(typ.toString))),
      state.n match {
        case None => ")"
        case Some(nested) => nest(2, concat(line, prettyAggStateSignatures(nested), ")"))
      }))
  }

  def prettyPhysicalAggSigs(aggSigs: Seq[PhysicalAggSig]): Doc =
    list(aggSigs.view.map(prettyPhysicalAggSig))

  def prettyPhysicalAggSig(aggSig: PhysicalAggSig): Doc = {
    nest(2, group(concat(
      "(",
      aggSig match {
        case GroupedAggSig(t, nested) =>
          concat("Grouped ", t.toString, line, prettyPhysicalAggSigs(nested))
        case ArrayLenAggSig(kl, nested) =>
          concat("ArrayLen ", prettyBooleanLiteral(kl), line, prettyPhysicalAggSigs(nested))
        case AggElementsAggSig(nested) =>
          concat("AggElements", line, prettyPhysicalAggSigs(nested))
        case PhysicalAggSig(op, state) =>
          concat(prettyClass(op), line, prettyAggStateSignature(state))
      },
      ")")))
  }

  def parens(d: Doc): Doc = nest(2, group(concat("(", d, ")")))

  def header(ir: BaseIR, elideLiterals: Boolean): Iterable[Doc] = ir match {
    case ApplyAggOp(initOpArgs, seqOpArgs, aggSig) => FastSeq(prettyClass(aggSig.op))
    case ApplyScanOp(initOpArgs, seqOpArgs, aggSig) => FastSeq(prettyClass(aggSig.op))
    case InitOp(i, args, aggSig) => FastSeq(i.toString, prettyPhysicalAggSig(aggSig))
    case SeqOp(i, args, aggSig) => FastSeq(i.toString, prettyPhysicalAggSig(aggSig))
    case CombOp(i1, i2, aggSig) => FastSeq(i1.toString, i2.toString, prettyPhysicalAggSig(aggSig))
    case ResultOp(i, aggSigs) => FastSeq(i.toString, prettyPhysicalAggSigs(aggSigs))
    case AggStateValue(i, sig) => FastSeq(i.toString, prettyAggStateSignature(sig))
    case InitFromSerializedValue(i, value, aggSig) =>
      FastSeq(i.toString, prettyAggStateSignature(aggSig))
    case CombOpValue(i, value, sig) => FastSeq(i.toString, prettyPhysicalAggSig(sig))
    case SerializeAggs(i, i2, spec, aggSigs) =>
      FastSeq(i.toString, i2.toString, prettyStringLiteral(spec.toString), prettyAggStateSignatures(aggSigs))
    case DeserializeAggs(i, i2, spec, aggSigs) =>
      FastSeq(i.toString, i2.toString, prettyStringLiteral(spec.toString), prettyAggStateSignatures(aggSigs))
    case RunAgg(body, result, signature) => FastSeq(prettyAggStateSignatures(signature))
    case RunAggScan(a, name, init, seq, res, signature) =>
      FastSeq(prettyIdentifier(name), prettyAggStateSignatures(signature))
    case I32(x) => FastSeq(x.toString)
    case I64(x) => FastSeq(x.toString)
    case F32(x) => FastSeq(x.toString)
    case F64(x) => FastSeq(x.toString)
    case Str(x) => FastSeq(prettyStringLiteral(if (elideLiterals && x.length > 13) x.take(10) + "..." else x))
    case UUID4(id) => FastSeq(prettyIdentifier(id))
    case Cast(_, typ) => FastSeq(typ.parsableString())
    case CastRename(_, typ) => FastSeq(typ.parsableString())
    case NA(typ) => FastSeq(typ.parsableString())
    case Literal(typ, value) =>
      FastSeq(typ.parsableString(),
        if (!elideLiterals)
          prettyStringLiteral(JsonMethods.compact(JSONAnnotationImpex.exportAnnotation(value, typ)))
        else
          "<literal value>")
    case EncodedLiteral(codec, _) => FastSeq(codec.encodedVirtualType.parsableString())
    case Let(name, _, _) => FastSeq(prettyIdentifier(name))
    case AggLet(name, _, _, isScan) => FastSeq(prettyIdentifier(name), prettyBooleanLiteral(isScan))
    case TailLoop(name, args, _) => FastSeq(prettyIdentifier(name), prettyIdentifiers(args.map(_._1).toFastIndexedSeq))
    case Recur(name, _, t) => FastSeq(prettyIdentifier(name), t.parsableString())
    // case Ref(name, t) => prettyIdentifier(name) + Option(t).map(x => s" $x").getOrElse("")  // For debug purposes
    case Ref(name, _) => FastSeq(prettyIdentifier(name))
    case RelationalRef(name, t) => FastSeq(prettyIdentifier(name), t.parsableString())
    case RelationalLet(name, _, _) => FastSeq(prettyIdentifier(name))
    case ApplyBinaryPrimOp(op, _, _) => FastSeq(prettyClass(op))
    case ApplyUnaryPrimOp(op, _) => FastSeq(prettyClass(op))
    case ApplyComparisonOp(op, _, _) => FastSeq(op.render())
    case GetField(_, name) => FastSeq(prettyIdentifier(name))
    case GetTupleElement(_, idx) => FastSeq(idx.toString)
    case MakeTuple(fields) => FastSeq(prettyInts(fields.map(_._1).toFastIndexedSeq, elideLiterals))
    case MakeArray(_, typ) => FastSeq(typ.parsableString())
    case MakeStream(_, typ, separateRegions) =>
      FastSeq(typ.parsableString(), prettyBooleanLiteral(separateRegions))
    case StreamRange(_, _, _, separateRegions) => FastSeq(prettyBooleanLiteral(separateRegions))
    case ToStream(_, separateRegions) => FastSeq(prettyBooleanLiteral(separateRegions))
    case StreamMap(_, name, _) => FastSeq(prettyIdentifier(name))
    case StreamMerge(_, _, key) => FastSeq(prettyIdentifiers(key))
    case StreamZip(_, names, _, behavior) => FastSeq(behavior match {
      case ArrayZipBehavior.AssertSameLength => "AssertSameLength"
      case ArrayZipBehavior.TakeMinLength => "TakeMinLength"
      case ArrayZipBehavior.ExtendNA => "ExtendNA"
      case ArrayZipBehavior.AssumeSameLength => "AssumeSameLength"
    }, prettyIdentifiers(names))
    case StreamZipJoin(_, key, curKey, curVals, _) =>
      FastSeq(prettyIdentifiers(key), prettyIdentifier(curKey), prettyIdentifier(curVals))
    case StreamMultiMerge(_, key) => FastSeq(prettyIdentifiers(key))
    case StreamFilter(_, name, _) => FastSeq(prettyIdentifier(name))
    case StreamFlatMap(_, name, _) => FastSeq(prettyIdentifier(name))
    case StreamFold(_, _, accumName, valueName, _) => FastSeq(prettyIdentifier(accumName), prettyIdentifier(valueName))
    case StreamFold2(_, acc, valueName, _, _) => FastSeq(prettyIdentifiers(acc.map(_._1)), prettyIdentifier(valueName))
    case StreamScan(_, _, accumName, valueName, _) => FastSeq(prettyIdentifier(accumName), prettyIdentifier(valueName))
    case StreamJoinRightDistinct(_, _, lKey, rKey, l, r, _, joinType) =>
      FastSeq(prettyIdentifiers(lKey), prettyIdentifiers(rKey), prettyIdentifier(l), prettyIdentifier(r), joinType)
    case StreamFor(_, valueName, _) => FastSeq(prettyIdentifier(valueName))
    case StreamAgg(a, name, query) => FastSeq(prettyIdentifier(name))
    case StreamAggScan(a, name, query) => FastSeq(prettyIdentifier(name))
    case AggExplode(_, name, _, isScan) => FastSeq(prettyIdentifier(name), prettyBooleanLiteral(isScan))
    case AggFilter(_, _, isScan) => FastSeq(prettyBooleanLiteral(isScan))
    case AggGroupBy(_, _, isScan) => FastSeq(prettyBooleanLiteral(isScan))
    case AggArrayPerElement(_, elementName, indexName, _, knownLength, isScan) =>
      FastSeq(prettyIdentifier(elementName), prettyIdentifier(indexName), prettyBooleanLiteral(isScan), prettyBooleanLiteral(knownLength.isDefined))
    case NDArrayMap(_, name, _) => FastSeq(prettyIdentifier(name))
    case NDArrayMap2(_, _, lName, rName, _) => FastSeq(prettyIdentifier(lName), prettyIdentifier(rName))
    case NDArrayReindex(_, indexExpr) => FastSeq(prettyInts(indexExpr, elideLiterals))
    case NDArrayConcat(_, axis) => FastSeq(axis.toString)
    case NDArrayAgg(_, axes) => FastSeq(prettyInts(axes, elideLiterals))
    case NDArrayRef(_, _, errorId) => FastSeq(s"$errorId")
    case ArraySort(_, l, r, _) => FastSeq(prettyIdentifier(l), prettyIdentifier(r))
    case ApplyIR(function, typeArgs, _) => FastSeq(prettyIdentifier(function), prettyTypes(typeArgs), ir.typ.parsableString())
    case Apply(function, typeArgs, _, t) => FastSeq(prettyIdentifier(function), prettyTypes(typeArgs), t.parsableString())
    case ApplySeeded(function, _, seed, t) => FastSeq(prettyIdentifier(function), seed.toString, t.parsableString())
    case ApplySpecial(function, typeArgs, _, t) => FastSeq(prettyIdentifier(function), prettyTypes(typeArgs), t.parsableString())
    case SelectFields(_, fields) => fields.view.map(f => text(prettyIdentifier(f))).intersperse[Doc]("(", space, ")")
    case LowerBoundOnOrderedCollection(_, _, onKey) => FastSeq(prettyBooleanLiteral(onKey))
    case In(i, typ) => FastSeq(text(s"$typ $i"))
    case Die(message, typ, errorId) => FastSeq(typ.parsableString(), errorId.toString)
    case CollectDistributedArray(_, _, cname, gname, _) =>
      FastSeq(prettyIdentifier(cname), prettyIdentifier(gname))
    case MatrixRead(typ, dropCols, dropRows, reader) =>
      FastSeq(if (typ == reader.fullMatrixType) "None" else typ.parsableString(),
        prettyBooleanLiteral(dropCols),
        prettyBooleanLiteral(dropRows),
        '"' + StringEscapeUtils.escapeString(JsonMethods.compact(reader.toJValue)) + '"')
    case MatrixWrite(_, writer) =>
      FastSeq('"' + StringEscapeUtils.escapeString(Serialization.write(writer)(MatrixWriter.formats)) + '"')
    case MatrixMultiWrite(_, writer) =>
      FastSeq('"' + StringEscapeUtils.escapeString(Serialization.write(writer)(MatrixNativeMultiWriter.formats)) + '"')
    case BlockMatrixRead(reader) =>
      FastSeq('"' + StringEscapeUtils.escapeString(JsonMethods.compact(reader.toJValue)) + '"')
    case BlockMatrixWrite(_, writer) =>
      FastSeq('"' + StringEscapeUtils.escapeString(Serialization.write(writer)(BlockMatrixWriter.formats)) + '"')
    case BlockMatrixMultiWrite(_, writer) =>
      FastSeq('"' + StringEscapeUtils.escapeString(Serialization.write(writer)(BlockMatrixWriter.formats)) + '"')
    case BlockMatrixBroadcast(_, inIndexExpr, shape, blockSize) =>
      FastSeq(prettyInts(inIndexExpr, elideLiterals),
        prettyLongs(shape, elideLiterals),
        blockSize.toString)
    case BlockMatrixAgg(_, outIndexExpr) => FastSeq(prettyInts(outIndexExpr, elideLiterals))
    case BlockMatrixSlice(_, slices) =>
      FastSeq(concat(slices.view.map(slice => prettyLongs(slice, elideLiterals)).intersperse[Doc]("(", space, ")")))
    case ValueToBlockMatrix(_, shape, blockSize) =>
      FastSeq(prettyLongs(shape, elideLiterals), blockSize.toString)
    case BlockMatrixFilter(_, indicesToKeepPerDim) =>
      FastSeq(concat(indicesToKeepPerDim.toSeq.view.map(indices => prettyLongs(indices, elideLiterals)).intersperse[Doc]("(", space, ")")))
    case BlockMatrixSparsify(_, sparsifier) =>
      FastSeq(sparsifier.pretty())
    case BlockMatrixRandom(seed, gaussian, shape, blockSize) =>
      FastSeq(seed.toString,
        prettyBooleanLiteral(gaussian),
        prettyLongs(shape, elideLiterals),
        blockSize.toString)
    case BlockMatrixMap(_, name, _, needsDense) =>
      FastSeq(prettyIdentifier(name), prettyBooleanLiteral(needsDense))
    case BlockMatrixMap2(_, _, lName, rName, _, sparsityStrategy) =>
      FastSeq(prettyIdentifier(lName), prettyIdentifier(rName), prettyClass(sparsityStrategy))
    case MatrixRowsHead(_, n) => FastSeq(n.toString)
    case MatrixColsHead(_, n) => FastSeq(n.toString)
    case MatrixRowsTail(_, n) => FastSeq(n.toString)
    case MatrixColsTail(_, n) => FastSeq(n.toString)
    case MatrixAnnotateRowsTable(_, _, uid, product) =>
      FastSeq(prettyStringLiteral(uid), prettyBooleanLiteral(product))
    case MatrixAnnotateColsTable(_, _, uid) => FastSeq(prettyStringLiteral(uid))
    case MatrixExplodeRows(_, path) => FastSeq(prettyIdentifiers(path))
    case MatrixExplodeCols(_, path) => FastSeq(prettyIdentifiers(path))
    case MatrixRepartition(_, n, strategy) => FastSeq(s"$n $strategy")
    case MatrixChooseCols(_, oldIndices) => FastSeq(prettyInts(oldIndices, elideLiterals))
    case MatrixMapCols(_, _, newKey) => FastSeq(prettyStringsOpt(newKey))
    case MatrixKeyRowsBy(_, keys, isSorted) =>
      FastSeq(prettyIdentifiers(keys), prettyBooleanLiteral(isSorted))
    case TableRead(typ, dropRows, tr) =>
      FastSeq(if (typ == tr.fullType) "None" else typ.parsableString(),
        prettyBooleanLiteral(dropRows),
        '"' + StringEscapeUtils.escapeString(JsonMethods.compact(tr.toJValue)) + '"')
    case TableWrite(_, writer) =>
      FastSeq('"' + StringEscapeUtils.escapeString(Serialization.write(writer)(TableWriter.formats)) + '"')
    case TableMultiWrite(_, writer) =>
      FastSeq('"' + StringEscapeUtils.escapeString(Serialization.write(writer)(WrappedMatrixNativeMultiWriter.formats)) + '"')
    case TableKeyBy(_, keys, isSorted) =>
      FastSeq(prettyIdentifiers(keys), prettyBooleanLiteral(isSorted))
    case TableRange(n, nPartitions) => FastSeq(n.toString, nPartitions.toString)
    case TableRepartition(_, n, strategy) => FastSeq(n.toString, strategy.toString)
    case TableHead(_, n) => FastSeq(n.toString)
    case TableTail(_, n) => FastSeq(n.toString)
    case TableJoin(_, _, joinType, joinKey) => FastSeq(joinType, joinKey.toString)
    case TableLeftJoinRightDistinct(_, _, root) => FastSeq(prettyIdentifier(root))
    case TableIntervalJoin(_, _, root, product) =>
      FastSeq(prettyIdentifier(root), prettyBooleanLiteral(product))
    case TableMultiWayZipJoin(_, dataName, globalName) =>
      FastSeq(prettyStringLiteral(dataName), prettyStringLiteral(globalName))
    case TableKeyByAndAggregate(_, _, _, nPartitions, bufferSize) =>
      FastSeq(prettyIntOpt(nPartitions), bufferSize.toString)
    case TableExplode(_, path) => FastSeq(prettyStrings(path))
    case TableMapPartitions(_, g, p, _) => FastSeq(prettyIdentifier(g), prettyIdentifier(p))
    case TableParallelize(_, nPartitions) => FastSeq(prettyIntOpt(nPartitions))
    case TableOrderBy(_, sortFields) => FastSeq(prettySortFields(sortFields))
    case CastMatrixToTable(_, entriesFieldName, colsFieldName) =>
      FastSeq(prettyStringLiteral(entriesFieldName), prettyStringLiteral(colsFieldName))
    case CastTableToMatrix(_, entriesFieldName, colsFieldName, colKey) =>
      FastSeq(prettyIdentifier(entriesFieldName), prettyIdentifier(colsFieldName), prettyIdentifiers(colKey))
    case MatrixToMatrixApply(_, function) =>
      FastSeq(prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats)))
    case MatrixToTableApply(_, function) =>
      FastSeq(prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats)))
    case TableToTableApply(_, function) =>
      FastSeq(prettyStringLiteral(JsonMethods.compact(function.toJValue)))
    case TableToValueApply(_, function) =>
      FastSeq(prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats)))
    case MatrixToValueApply(_, function) =>
      FastSeq(prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats)))
    case BlockMatrixToTableApply(_, _, function) =>
      FastSeq(prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats)))
    case TableRename(_, rowMap, globalMap) =>
      val rowKV = rowMap.toArray
      val globalKV = globalMap.toArray
      FastSeq(prettyStrings(rowKV.map(_._1)), prettyStrings(rowKV.map(_._2)),
        prettyStrings(globalKV.map(_._1)), prettyStrings(globalKV.map(_._2)))
    case MatrixRename(_, globalMap, colMap, rowMap, entryMap) =>
      val globalKV = globalMap.toArray
      val colKV = colMap.toArray
      val rowKV = rowMap.toArray
      val entryKV = entryMap.toArray
      FastSeq(prettyStrings(globalKV.map(_._1)), prettyStrings(globalKV.map(_._2)),
        prettyStrings(colKV.map(_._1)), prettyStrings(colKV.map(_._2)),
        prettyStrings(rowKV.map(_._1)), prettyStrings(rowKV.map(_._2)),
        prettyStrings(entryKV.map(_._1)), prettyStrings(entryKV.map(_._2)))
    case TableFilterIntervals(child, intervals, keep) =>
      FastSeq(
        prettyStringLiteral(Serialization.write(
          JSONAnnotationImpex.exportAnnotation(intervals, TArray(TInterval(child.typ.keyType)))
        )(RelationalSpec.formats)),
        prettyBooleanLiteral(keep))
    case MatrixFilterIntervals(child, intervals, keep) =>
      FastSeq(
        prettyStringLiteral(Serialization.write(
          JSONAnnotationImpex.exportAnnotation(intervals, TArray(TInterval(child.typ.rowKeyStruct)))
        )(RelationalSpec.formats)),
        prettyBooleanLiteral(keep))
    case RelationalLetTable(name, _, _) => FastSeq(prettyIdentifier(name))
    case RelationalLetMatrixTable(name, _, _) => FastSeq(prettyIdentifier(name))
    case RelationalLetBlockMatrix(name, _, _) => FastSeq(prettyIdentifier(name))
    case ReadPartition(_, rowType, reader) =>
      FastSeq(rowType.parsableString(),
           prettyStringLiteral(JsonMethods.compact(reader.toJValue)))
    case WritePartition(value, writeCtx, writer) =>
      FastSeq(prettyStringLiteral(JsonMethods.compact(writer.toJValue)))
    case WriteMetadata(writeAnnotations, writer) =>
      FastSeq(prettyStringLiteral(JsonMethods.compact(writer.toJValue)))
    case ReadValue(_, spec, reqType) =>
      FastSeq(prettyStringLiteral(spec.toString), reqType.parsableString())
    case WriteValue(_, _, spec) => FastSeq(prettyStringLiteral(spec.toString))
    case x@ShuffleWith(_, _, _, _, name, _, _) =>
      FastSeq(x.shuffleType.parsableString(), prettyIdentifier(name))

    case _ => FastSeq()
  }

  def apply(ir: BaseIR, width: Int = 80, ribbonWidth: Int = 80, elideLiterals: Boolean = true, maxLen: Int = -1): String = {
    def prettySeq(xs: Seq[BaseIR]): Doc =
      list(xs.view.map(pretty))

    def pretty(ir: BaseIR): Doc = {

      val body: Iterable[Doc] = ir match {
        case MakeStruct(fields) =>
          fields.view.map { case (n, a) =>
            parens(concat(n, line, pretty(a)))
          }
        case ApplyAggOp(initOpArgs, seqOpArgs, aggSig) =>
          FastSeq(prettySeq(initOpArgs), prettySeq(seqOpArgs))
        case ApplyScanOp(initOpArgs, seqOpArgs, aggSig) =>
          FastSeq(prettySeq(initOpArgs), prettySeq(seqOpArgs))
        case InitOp(i, args, aggSig) => FastSeq(prettySeq(args))
        case SeqOp(i, args, aggSig) => FastSeq(prettySeq(args))
        case InsertFields(old, fields, fieldOrder) =>
          val fieldDocs = fields.view.map { case (n, a) =>
            list(prettyIdentifier(n), pretty(a))
          }
          Iterable.concat(FastSeq(pretty(old), prettyStringsOpt(fieldOrder)), fieldDocs)
        case _ => ir.children.view.map(pretty)
      }

      /*
      val pt = ir match{
        case ir: IR => if (ir._pType != null) concat(space, ir.pType.toString)
        case _ => empty
      }
      nest(2, group(concat("(", prettyClass(ir), pt, body, ")")))
      */

      nest(2,
        group(
          concat(
            "(",
            vsep(Iterable.concat(
              FastSeq(nest(2, concat(prettyClass(ir), fillSep(header(ir, elideLiterals))))),
              body)),
            ")")))
    }

    pretty(ir).render(width, ribbonWidth)
  }
}
