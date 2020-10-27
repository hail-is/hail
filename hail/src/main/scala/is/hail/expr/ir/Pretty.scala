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

  def apply(ir: BaseIR, width: Int = 80, elideLiterals: Boolean = true, maxLen: Int = -1): String = {
    def prettySeq(xs: Seq[BaseIR]): Doc =
      list(xs.view.map(pretty))

    def pretty(ir: BaseIR): Doc = {
      val body = ir match {
        case MakeStruct(fields) =>
          vsep(fields.view.map { case (n, a) =>
            parens(concat(n, line, pretty(a)))
          })
        case ApplyAggOp(initOpArgs, seqOpArgs, aggSig) =>
          vsep(prettyClass(aggSig.op), prettySeq(initOpArgs), prettySeq(seqOpArgs))
        case ApplyScanOp(initOpArgs, seqOpArgs, aggSig) =>
          vsep(prettyClass(aggSig.op), prettySeq(initOpArgs), prettySeq(seqOpArgs))
        case InitOp(i, args, aggSig) =>
          vsep(i.toString, prettyPhysicalAggSig(aggSig), prettySeq(args))
        case SeqOp(i, args, aggSig) =>
          vsep(i.toString, prettyPhysicalAggSig(aggSig), prettySeq(args))
        case CombOp(i1, i2, aggSig) =>
          concat(i1.toString, space, i2.toString, line, prettyPhysicalAggSig(aggSig))
        case ResultOp(i, aggSigs) =>
          concat(i.toString, line, prettyPhysicalAggSigs(aggSigs))
        case AggStateValue(i, sig) =>
          concat(i.toString, line, prettyAggStateSignature(sig))
        case InitFromSerializedValue(i, value, aggSig) =>
          vsep(i.toString, prettyAggStateSignature(aggSig), pretty(value))
        case CombOpValue(i, value, sig) =>
          vsep(i.toString, prettyPhysicalAggSig(sig), pretty(value))
        case SerializeAggs(i, i2, spec, aggSigs) =>
          concat(i.toString, space, i2.toString, space, prettyStringLiteral(spec.toString), line, prettyAggStateSignatures(aggSigs))
        case DeserializeAggs(i, i2, spec, aggSigs) =>
          concat(i.toString, space, i2.toString, space, prettyStringLiteral(spec.toString), line, prettyAggStateSignatures(aggSigs))
        case RunAgg(body, result, signature) =>
          vsep(prettyAggStateSignatures(signature), pretty(body), pretty(result))
        case RunAggScan(a, name, init, seq, res, signature) =>
          vsep(prettyIdentifier(name), prettyAggStateSignatures(signature), pretty(a), pretty(init), pretty(seq), pretty(res))
        case InsertFields(old, fields, fieldOrder) =>
          val fieldDocs = fields.view.map { case (n, a) =>
            list(prettyIdentifier(n), pretty(a))
          }
          vsep(Iterable.concat(Array(pretty(old), prettyStringsOpt(fieldOrder)), fieldDocs))
        case _ =>
          val header: Doc = ir match {
            case I32(x) => x.toString
            case I64(x) => x.toString
            case F32(x) => x.toString
            case F64(x) => x.toString
            case Str(x) => prettyStringLiteral(if (elideLiterals && x.length > 13) x.take(10) + "..." else x)
            case UUID4(id) => prettyIdentifier(id)
            case Cast(_, typ) => typ.parsableString()
            case CastRename(_, typ) => typ.parsableString()
            case NA(typ) => typ.parsableString()
            case Literal(typ, value) =>
              hsep(typ.parsableString(),
                if (!elideLiterals)
                  prettyStringLiteral(JsonMethods.compact(JSONAnnotationImpex.exportAnnotation(value, typ)))
                else
                  "<literal value>")
            case EncodedLiteral(codec, _) => codec.encodedVirtualType.parsableString()
            case Let(name, _, _) => prettyIdentifier(name)
            case AggLet(name, _, _, isScan) => hsep(prettyIdentifier(name), prettyBooleanLiteral(isScan))
            case TailLoop(name, args, _) => hsep(prettyIdentifier(name), prettyIdentifiers(args.map(_._1).toFastIndexedSeq))
            case Recur(name, _, t) => hsep(prettyIdentifier(name), t.parsableString())
            // case Ref(name, t) => prettyIdentifier(name) + Option(t).map(x => s" $x").getOrElse("")  // For debug purposes
            case Ref(name, _) => prettyIdentifier(name)
            case RelationalRef(name, t) => hsep(prettyIdentifier(name), t.parsableString())
            case RelationalLet(name, _, _) => prettyIdentifier(name)
            case ApplyBinaryPrimOp(op, _, _) => prettyClass(op)
            case ApplyUnaryPrimOp(op, _) => prettyClass(op)
            case ApplyComparisonOp(op, _, _) => op.render()
            case GetField(_, name) => prettyIdentifier(name)
            case GetTupleElement(_, idx) => idx.toString
            case MakeTuple(fields) => prettyInts(fields.map(_._1).toFastIndexedSeq, elideLiterals)
            case MakeArray(_, typ) => typ.parsableString()
            case MakeStream(_, typ, separateRegions) =>
              hsep(typ.parsableString(), prettyBooleanLiteral(separateRegions))
            case StreamRange(_, _, _, separateRegions) => prettyBooleanLiteral(separateRegions)
            case ToStream(_, separateRegions) => prettyBooleanLiteral(separateRegions)
            case StreamMap(_, name, _) => prettyIdentifier(name)
            case StreamMerge(_, _, key) => prettyIdentifiers(key)
            case StreamZip(_, names, _, behavior) => hsep(behavior match {
              case ArrayZipBehavior.AssertSameLength => "AssertSameLength"
              case ArrayZipBehavior.TakeMinLength => "TakeMinLength"
              case ArrayZipBehavior.ExtendNA => "ExtendNA"
              case ArrayZipBehavior.AssumeSameLength => "AssumeSameLength"
            }, prettyIdentifiers(names))
            case StreamZipJoin(_, key, curKey, curVals, _) =>
              hsep(prettyIdentifiers(key), prettyIdentifier(curKey), prettyIdentifier(curVals))
            case StreamMultiMerge(_, key) => prettyIdentifiers(key)
            case StreamFilter(_, name, _) => prettyIdentifier(name)
            case StreamFlatMap(_, name, _) => prettyIdentifier(name)
            case StreamFold(_, _, accumName, valueName, _) => hsep(prettyIdentifier(accumName), prettyIdentifier(valueName))
            case StreamFold2(_, acc, valueName, _, _) => hsep(prettyIdentifiers(acc.map(_._1)), prettyIdentifier(valueName))
            case StreamScan(_, _, accumName, valueName, _) => hsep(prettyIdentifier(accumName), prettyIdentifier(valueName))
            case StreamJoinRightDistinct(_, _, lKey, rKey, l, r, _, joinType) =>
              hsep(prettyIdentifiers(lKey), prettyIdentifiers(rKey), prettyIdentifier(l), prettyIdentifier(r), joinType)
            case StreamFor(_, valueName, _) => prettyIdentifier(valueName)
            case StreamAgg(a, name, query) => prettyIdentifier(name)
            case StreamAggScan(a, name, query) => prettyIdentifier(name)
            case AggExplode(_, name, _, isScan) => hsep(prettyIdentifier(name), prettyBooleanLiteral(isScan))
            case AggFilter(_, _, isScan) => prettyBooleanLiteral(isScan)
            case AggGroupBy(_, _, isScan) => prettyBooleanLiteral(isScan)
            case AggArrayPerElement(_, elementName, indexName, _, knownLength, isScan) =>
              hsep(prettyIdentifier(elementName), prettyIdentifier(indexName), prettyBooleanLiteral(isScan), prettyBooleanLiteral(knownLength.isDefined))
            case NDArrayMap(_, name, _) => prettyIdentifier(name)
            case NDArrayMap2(_, _, lName, rName, _) => hsep(prettyIdentifier(lName), prettyIdentifier(rName))
            case NDArrayReindex(_, indexExpr) => prettyInts(indexExpr, elideLiterals)
            case NDArrayConcat(_, axis) => axis.toString
            case NDArrayAgg(_, axes) => prettyInts(axes, elideLiterals)
            case NDArrayRef(_, _, errorId) => s"$errorId"
            case ArraySort(_, l, r, _) => hsep(prettyIdentifier(l), prettyIdentifier(r))
            case ApplyIR(function, typeArgs, _) => hsep(prettyIdentifier(function), prettyTypes(typeArgs), ir.typ.parsableString())
            case Apply(function, typeArgs, _, t) => hsep(prettyIdentifier(function), prettyTypes(typeArgs), t.parsableString())
            case ApplySeeded(function, _, seed, t) => hsep(prettyIdentifier(function), seed.toString, t.parsableString())
            case ApplySpecial(function, typeArgs, _, t) => hsep(prettyIdentifier(function), prettyTypes(typeArgs), t.parsableString())
            case SelectFields(_, fields) => concat(fields.view.map(f => text(prettyIdentifier(f))).intersperse[Doc]("(", space, ")"))
            case LowerBoundOnOrderedCollection(_, _, onKey) => prettyBooleanLiteral(onKey)
            case In(i, typ) => s"$typ $i"
            case Die(message, typ, errorId) => hsep(typ.parsableString(), errorId.toString)
            case CollectDistributedArray(_, _, cname, gname, _) =>
              hsep(prettyIdentifier(cname), prettyIdentifier(gname))
            case MatrixRead(typ, dropCols, dropRows, reader) =>
              hsep(if (typ == reader.fullMatrixType) "None" else typ.parsableString(),
                prettyBooleanLiteral(dropCols),
                prettyBooleanLiteral(dropRows),
                '"' + StringEscapeUtils.escapeString(JsonMethods.compact(reader.toJValue)) + '"')
            case MatrixWrite(_, writer) =>
              '"' + StringEscapeUtils.escapeString(Serialization.write(writer)(MatrixWriter.formats)) + '"'
            case MatrixMultiWrite(_, writer) =>
              '"' + StringEscapeUtils.escapeString(Serialization.write(writer)(MatrixNativeMultiWriter.formats)) + '"'
            case BlockMatrixRead(reader) =>
              '"' + StringEscapeUtils.escapeString(JsonMethods.compact(reader.toJValue)) + '"'
            case BlockMatrixWrite(_, writer) =>
              '"' + StringEscapeUtils.escapeString(Serialization.write(writer)(BlockMatrixWriter.formats)) + '"'
            case BlockMatrixMultiWrite(_, writer) =>
              '"' + StringEscapeUtils.escapeString(Serialization.write(writer)(BlockMatrixWriter.formats)) + '"'
            case BlockMatrixBroadcast(_, inIndexExpr, shape, blockSize) =>
              hsep(prettyInts(inIndexExpr, elideLiterals),
                prettyLongs(shape, elideLiterals),
                blockSize.toString)
            case BlockMatrixAgg(_, outIndexExpr) => prettyInts(outIndexExpr, elideLiterals)
            case BlockMatrixSlice(_, slices) =>
              concat(slices.view.map(slice => prettyLongs(slice, elideLiterals)).intersperse[Doc]("(", space, ")"))
            case ValueToBlockMatrix(_, shape, blockSize) =>
              hsep(prettyLongs(shape, elideLiterals), blockSize.toString)
            case BlockMatrixFilter(_, indicesToKeepPerDim) =>
              concat(indicesToKeepPerDim.toSeq.view.map(indices => prettyLongs(indices, elideLiterals)).intersperse[Doc]("(", space, ")"))
            case BlockMatrixSparsify(_, sparsifier) =>
              sparsifier.pretty()
            case BlockMatrixRandom(seed, gaussian, shape, blockSize) =>
              hsep(seed.toString,
                prettyBooleanLiteral(gaussian),
                prettyLongs(shape, elideLiterals),
                blockSize.toString)
            case BlockMatrixMap(_, name, _, needsDense) =>
              hsep(prettyIdentifier(name), prettyBooleanLiteral(needsDense))
            case BlockMatrixMap2(_, _, lName, rName, _, sparsityStrategy) =>
              hsep(prettyIdentifier(lName), prettyIdentifier(rName), prettyClass(sparsityStrategy))
            case MatrixRowsHead(_, n) => n.toString
            case MatrixColsHead(_, n) => n.toString
            case MatrixRowsTail(_, n) => n.toString
            case MatrixColsTail(_, n) => n.toString
            case MatrixAnnotateRowsTable(_, _, uid, product) =>
              hsep(prettyStringLiteral(uid), prettyBooleanLiteral(product))
            case MatrixAnnotateColsTable(_, _, uid) => prettyStringLiteral(uid)
            case MatrixExplodeRows(_, path) => prettyIdentifiers(path)
            case MatrixExplodeCols(_, path) => prettyIdentifiers(path)
            case MatrixRepartition(_, n, strategy) => s"$n $strategy"
            case MatrixChooseCols(_, oldIndices) => prettyInts(oldIndices, elideLiterals)
            case MatrixMapCols(_, _, newKey) => prettyStringsOpt(newKey)
            case MatrixKeyRowsBy(_, keys, isSorted) =>
              hsep(prettyIdentifiers(keys), prettyBooleanLiteral(isSorted))
            case TableRead(typ, dropRows, tr) =>
              hsep(if (typ == tr.fullType) "None" else typ.parsableString(),
                prettyBooleanLiteral(dropRows),
                '"' + StringEscapeUtils.escapeString(JsonMethods.compact(tr.toJValue)) + '"')
            case TableWrite(_, writer) =>
              '"' + StringEscapeUtils.escapeString(Serialization.write(writer)(TableWriter.formats)) + '"'
            case TableMultiWrite(_, writer) =>
              '"' + StringEscapeUtils.escapeString(Serialization.write(writer)(WrappedMatrixNativeMultiWriter.formats)) + '"'
            case TableKeyBy(_, keys, isSorted) =>
              hsep(prettyIdentifiers(keys), prettyBooleanLiteral(isSorted))
            case TableRange(n, nPartitions) => hsep(n.toString, nPartitions.toString)
            case TableRepartition(_, n, strategy) => hsep(n.toString, strategy.toString)
            case TableHead(_, n) => n.toString
            case TableTail(_, n) => n.toString
            case TableJoin(_, _, joinType, joinKey) => hsep(joinType, joinKey.toString)
            case TableLeftJoinRightDistinct(_, _, root) => prettyIdentifier(root)
            case TableIntervalJoin(_, _, root, product) =>
              hsep(prettyIdentifier(root), prettyBooleanLiteral(product))
            case TableMultiWayZipJoin(_, dataName, globalName) =>
              hsep(prettyStringLiteral(dataName), prettyStringLiteral(globalName))
            case TableKeyByAndAggregate(_, _, _, nPartitions, bufferSize) =>
              hsep(prettyIntOpt(nPartitions), bufferSize.toString)
            case TableExplode(_, path) => prettyStrings(path)
            case TableMapPartitions(_, g, p, _) => hsep(prettyIdentifier(g), prettyIdentifier(p))
            case TableParallelize(_, nPartitions) => prettyIntOpt(nPartitions)
            case TableOrderBy(_, sortFields) => prettySortFields(sortFields)
            case CastMatrixToTable(_, entriesFieldName, colsFieldName) =>
              hsep(prettyStringLiteral(entriesFieldName), prettyStringLiteral(colsFieldName))
            case CastTableToMatrix(_, entriesFieldName, colsFieldName, colKey) =>
              hsep(prettyIdentifier(entriesFieldName), prettyIdentifier(colsFieldName), prettyIdentifiers(colKey))
            case MatrixToMatrixApply(_, function) =>
              prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats))
            case MatrixToTableApply(_, function) =>
              prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats))
            case TableToTableApply(_, function) =>
              prettyStringLiteral(JsonMethods.compact(function.toJValue))
            case TableToValueApply(_, function) =>
              prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats))
            case MatrixToValueApply(_, function) =>
              prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats))
            case BlockMatrixToTableApply(_, _, function) =>
              prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats))
            case TableRename(_, rowMap, globalMap) =>
              val rowKV = rowMap.toArray
              val globalKV = globalMap.toArray
              hsep(prettyStrings(rowKV.map(_._1)), prettyStrings(rowKV.map(_._2)),
                prettyStrings(globalKV.map(_._1)), prettyStrings(globalKV.map(_._2)))
            case MatrixRename(_, globalMap, colMap, rowMap, entryMap) =>
              val globalKV = globalMap.toArray
              val colKV = colMap.toArray
              val rowKV = rowMap.toArray
              val entryKV = entryMap.toArray
              hsep(prettyStrings(globalKV.map(_._1)), prettyStrings(globalKV.map(_._2)),
                prettyStrings(colKV.map(_._1)), prettyStrings(colKV.map(_._2)),
                prettyStrings(rowKV.map(_._1)), prettyStrings(rowKV.map(_._2)),
                prettyStrings(entryKV.map(_._1)), prettyStrings(entryKV.map(_._2)))
            case TableFilterIntervals(child, intervals, keep) =>
              hsep(
                prettyStringLiteral(Serialization.write(
                  JSONAnnotationImpex.exportAnnotation(intervals, TArray(TInterval(child.typ.keyType)))
                )(RelationalSpec.formats)),
                prettyBooleanLiteral(keep))
            case MatrixFilterIntervals(child, intervals, keep) =>
              hsep(
                prettyStringLiteral(Serialization.write(
                  JSONAnnotationImpex.exportAnnotation(intervals, TArray(TInterval(child.typ.rowKeyStruct)))
                )(RelationalSpec.formats)),
                prettyBooleanLiteral(keep))
            case RelationalLetTable(name, _, _) => prettyIdentifier(name)
            case RelationalLetMatrixTable(name, _, _) => prettyIdentifier(name)
            case RelationalLetBlockMatrix(name, _, _) => prettyIdentifier(name)
            case ReadPartition(_, rowType, reader) =>
              hsep(rowType.parsableString(),
                   prettyStringLiteral(JsonMethods.compact(reader.toJValue)))
            case WritePartition(value, writeCtx, writer) =>
              prettyStringLiteral(JsonMethods.compact(writer.toJValue))
            case WriteMetadata(writeAnnotations, writer) =>
              prettyStringLiteral(JsonMethods.compact(writer.toJValue))
            case ReadValue(_, spec, reqType) =>
              hsep(prettyStringLiteral(spec.toString), reqType.parsableString())
            case WriteValue(_, _, spec) => prettyStringLiteral(spec.toString)
            case x@ShuffleWith(_, _, _, _, name, _, _) =>
              hsep(x.shuffleType.parsableString(), prettyIdentifier(name))

            case _ => empty
          }

          vsep(Iterable.concat(Array(header), ir.children.view.map(pretty)))
      }

      /*
      val pt = ir match{
        case ir: IR => if (ir._pType != null) concat(space, ir.pType.toString)
        case _ => empty
      }
      nest(2, group(concat("(", prettyClass(ir), pt, body, ")")))
      */

      nest(2, group(concat("(", prettyClass(ir), space, body, ")")))
    }

    pretty(ir).render(width)
  }
}
