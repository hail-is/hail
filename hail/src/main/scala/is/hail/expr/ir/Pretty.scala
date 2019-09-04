package is.hail.expr.ir

import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.functions.RelationalFunctions
import is.hail.expr.types.virtual.{TArray, TInterval}
import is.hail.table.Ascending
import is.hail.utils._
import is.hail.variant.RelationalSpec
import org.json4s.jackson.{JsonMethods, Serialization}

object Pretty {

  def short(ir: BaseIR, elideLiterals: Boolean = false, maxLen: Int = 100): String = {
    val s = Pretty(ir)
    if (s.length < maxLen) s else s.substring(0, maxLen)
  }

  def prettyStringLiteral(s: String): String =
    "\"" + StringEscapeUtils.escapeString(s) + "\""

  def prettyStrings(xs: IndexedSeq[String]): String = xs.map(prettyStringLiteral).mkString("(", " ", ")")

  def prettyStringsOpt(x: Option[IndexedSeq[String]]): String = x.map(prettyStrings).getOrElse("None")

  def prettyBooleanLiteral(b: Boolean): String =
    if (b) "True" else "False"

  def prettyClass(x: AnyRef): String =
    x.getClass.getName.split("\\.").last

  def prettyAggSignature(aggSig: AggSignature): String = {
    val sb = new StringBuilder
    sb += '('
    sb.append(prettyClass(aggSig.op))
    sb += ' '
    sb.append(aggSig.constructorArgs.map(_.parsableString()).mkString(" (", " ", ")"))
    sb.append(aggSig.initOpArgs.map(_.map(_.parsableString()).mkString(" (", " ", ")")).getOrElse(" None"))
    sb.append(aggSig.seqOpArgs.map(_.parsableString()).mkString(" (", " ", ")"))
    sb += ')'
    sb.result()
  }

  val MAX_VALUES_TO_LOG: Int = 25

  def apply(ir: BaseIR, elideLiterals: Boolean = false): String = {
    val sb = new StringBuilder

    def prettyIntOpt(x: Option[Int]): String = x.map(_.toString).getOrElse("None")

    def prettyLongs(x: IndexedSeq[Long]): String = if (elideLiterals && x.length > MAX_VALUES_TO_LOG)
      x.mkString("(", " ", s"... ${ x.length - MAX_VALUES_TO_LOG } more values... )")
    else
      x.mkString("(", " ", ")")

    def prettyInts(x: IndexedSeq[Int]): String = if (elideLiterals && x.length > MAX_VALUES_TO_LOG)
      x.mkString("(", " ", s"... ${ x.length - MAX_VALUES_TO_LOG } more values... )")
    else
      x.mkString("(", " ", ")")

    def prettyIdentifiers(x: IndexedSeq[String]): String = x.map(prettyIdentifier).mkString("(", " ", ")")

    def prettySeq(xs: Seq[BaseIR], depth: Int) {
      sb.append(" " * depth)
      sb += '('
      xs.foreach { x =>
        sb += '\n'
        pretty(x, depth + 2)
      }
      sb += ')'
    }

    def prettyAggSignature2(aggSig: AggSignature2, depth: Int): String = {
      sb.append(" " * depth)
      sb += '('
      sb.append(prettyClass(aggSig.op))
      sb += ' '
      sb.append(aggSig.initOpArgs.map(_.parsableString()).mkString(" (", " ", ")"))
      sb.append(aggSig.seqOpArgs.map(_.parsableString()).mkString(" (", " ", ")"))
      if (aggSig.nested.isEmpty)
        sb.append(" None")
      else
        aggSig.nested.foreach(prettyAggSeq(_, depth + 2))
      sb += ')'
      sb.result()
    }

    def prettyAggSeq(sigs: Seq[AggSignature2], depth: Int) {
      sb.append(" " * depth)
      sb += '('
      sigs.foreach { x =>
        sb += '\n'
        prettyAggSignature2(x, depth + 2)
      }
      sb += ')'
    }

    def pretty(ir: BaseIR, depth: Int) {
      sb.append(" " * depth)
      sb += '('

      sb.append(prettyClass(ir) )

      ir match {
        case MakeStruct(fields) =>
          if (fields.nonEmpty) {
            sb += '\n'
            fields.foreachBetween { case (n, a) =>
              sb.append(" " * (depth + 2))
              sb += '('
              sb.append(n)
              sb += '\n'
              pretty(a, depth + 4)
              sb += ')'
            }(sb += '\n')
          }
        case ApplyAggOp(ctorArgs, initOpArgs, seqOpArgs, aggSig) =>
          sb += ' '
          sb.append(prettyClass(aggSig.op))
          sb += '\n'
          prettySeq(ctorArgs, depth + 2)
          sb += '\n'
          initOpArgs match {
            case Some(initOpArgs) => prettySeq(initOpArgs, depth + 2)
            case None =>
              sb.append(" " * (depth + 2))
              sb.append("None")
          }
          sb += '\n'
          prettySeq(seqOpArgs, depth + 2)
        case ApplyScanOp(ctorArgs, initOpArgs, seqOpArgs, aggSig) =>
          sb += ' '
          sb.append(prettyClass(aggSig.op))
          sb += '\n'
          prettySeq(ctorArgs, depth + 2)
          sb += '\n'
          initOpArgs match {
            case Some(initOpArgs) => prettySeq(initOpArgs, depth + 2)
            case None =>
              sb.append(" " * (depth + 2))
              sb.append("None")
          }
          sb += '\n'
          prettySeq(seqOpArgs, depth + 2)
        case InitOp(i, args, aggSig) =>
          sb += ' '
          sb.append(prettyAggSignature(aggSig))
          sb += '\n'
          pretty(i, depth + 2)
          sb += '\n'
          prettySeq(args, depth + 2)
        case SeqOp(i, args, aggSig) =>
          sb += ' '
          sb.append(prettyAggSignature(aggSig))
          sb += '\n'
          pretty(i, depth + 2)
          sb += '\n'
          prettySeq(args, depth + 2)
        case InitOp2(i, args, aggSig) =>
          sb += ' '
          sb.append(i)
          sb += ' '
          prettyAggSignature2(aggSig, depth + 2)
          sb += '\n'
          prettySeq(args, depth + 2)
        case SeqOp2(i, args, aggSig) =>
          sb += ' '
          sb.append(i)
          sb += ' '
          prettyAggSignature2(aggSig, depth + 2)
          sb += '\n'
          prettySeq(args, depth + 2)
        case CombOp2(i1, i2, aggSig) =>
          sb += ' '
          sb.append(i1)
          sb += ' '
          sb.append(i2)
          sb += ' '
          prettyAggSignature2(aggSig, depth + 2)
        case ResultOp2(i, aggSigs) =>
          sb += ' '
          sb.append(i)
          sb += '\n'
          prettyAggSeq(aggSigs, depth + 2)
        case SerializeAggs(i, i2, spec, aggSigs) =>
          sb += ' '
          sb.append(i)
          sb += ' '
          sb.append(i2)
          sb += ' '
          sb.append(prettyStringLiteral(spec.toString))
          sb += '\n'
          prettyAggSeq(aggSigs, depth + 2)
        case DeserializeAggs(i, i2, spec, aggSigs) =>
          sb += ' '
          sb.append(i)
          sb += ' '
          sb.append(i2)
          sb += ' '
          sb.append(prettyStringLiteral(spec.toString))
          sb += '\n'
          prettyAggSeq(aggSigs, depth + 2)
        case InsertFields(old, fields, fieldOrder) =>
          sb += '\n'
          pretty(old, depth + 2)
          sb.append('\n')
          sb.append(" " * (depth + 2))
          sb.append(prettyStringsOpt(fieldOrder))
          if (fields.nonEmpty) {
            sb += '\n'
            fields.foreachBetween { case (n, a) =>
              sb.append(" " * (depth + 2))
              sb += '('
              sb.append(prettyIdentifier(n))
              sb += '\n'
              pretty(a, depth + 4)
              sb += ')'
            }(sb += '\n')
          }
        case _ =>
          val header = ir match {
            case I32(x) => x.toString
            case I64(x) => x.toString
            case F32(x) => x.toString
            case F64(x) => x.toString
            case Str(x) => prettyStringLiteral(x)
            case Cast(_, typ) => typ.parsableString()
            case CastRename(_, typ) => typ.parsableString()
            case NA(typ) => typ.parsableString()
            case Literal(typ, value) =>
              s"${ typ.parsableString() } " + (
                  if (!elideLiterals)
                    s"${ prettyStringLiteral(JsonMethods.compact(JSONAnnotationImpex.exportAnnotation(value, typ))) }"
                  else
                    "<literal value>"
                )
            case Let(name, _, _) => prettyIdentifier(name)
            case AggLet(name, _, _, isScan) => prettyIdentifier(name) + " " + prettyBooleanLiteral(isScan)
            case Ref(name, _) => prettyIdentifier(name)
            case RelationalRef(name, t) => prettyIdentifier(name) + " " + t.parsableString()
            case RelationalLet(name, _, _) => prettyIdentifier(name)
            case ApplyBinaryPrimOp(op, _, _) => prettyClass(op)
            case ApplyUnaryPrimOp(op, _) => prettyClass(op)
            case ApplyComparisonOp(op, _, _) => prettyClass(op)
            case GetField(_, name) => prettyIdentifier(name)
            case GetTupleElement(_, idx) => idx.toString
            case MakeTuple(fields) => prettyInts(fields.map(_._1).toFastIndexedSeq)
            case MakeArray(_, typ) => typ.parsableString()
            case MakeStream(_, typ) => typ.parsableString()
            case ArrayMap(_, name, _) => prettyIdentifier(name)
            case ArrayFilter(_, name, _) => prettyIdentifier(name)
            case ArrayFlatMap(_, name, _) => prettyIdentifier(name)
            case ArrayFold(_, _, accumName, valueName, _) => prettyIdentifier(accumName) + " " + prettyIdentifier(valueName)
            case ArrayScan(_, _, accumName, valueName, _) => prettyIdentifier(accumName) + " " + prettyIdentifier(valueName)
            case ArrayLeftJoinDistinct(_, _, l, r, _, _) => prettyIdentifier(l) + " " + prettyIdentifier(r)
            case ArrayFor(_, valueName, _) => prettyIdentifier(valueName)
            case ArrayAgg(a, name, query) => prettyIdentifier(name)
            case ArrayAggScan(a, name, query) => prettyIdentifier(name)
            case AggExplode(_, name, _, isScan) => prettyIdentifier(name) + " " + prettyBooleanLiteral(isScan)
            case AggFilter(_, _, isScan) => prettyBooleanLiteral(isScan)
            case AggGroupBy(_, _, isScan) => prettyBooleanLiteral(isScan)
            case AggArrayPerElement(_, elementName, indexName, _, knownLength, isScan) =>
              prettyIdentifier(elementName) + " " + prettyIdentifier(indexName) + " " + prettyBooleanLiteral(isScan) + " " + prettyBooleanLiteral(knownLength.isDefined)
            case NDArrayMap(_, name, _) => prettyIdentifier(name)
            case NDArrayMap2(_, _, lName, rName, _) => prettyIdentifier(lName) + " " + prettyIdentifier(rName)
            case NDArrayReindex(_, indexExpr) => prettyInts(indexExpr)
            case NDArrayAgg(_, axes) => prettyInts(axes)
            case ArraySort(_, l, r, _) => prettyIdentifier(l) + " " + prettyIdentifier(r)
            case ApplyIR(function, _) => prettyIdentifier(function) + " " + ir.typ.parsableString()
            case Apply(function, _, t) => prettyIdentifier(function) + " " + t.parsableString()
            case ApplySeeded(function, _, seed, t) => prettyIdentifier(function) + " " + seed.toString + " " + t.parsableString()
            case ApplySpecial(function, _, t) => prettyIdentifier(function) + " " + t.parsableString()
            case SelectFields(_, fields) => fields.map(prettyIdentifier).mkString("(", " ", ")")
            case LowerBoundOnOrderedCollection(_, _, onKey) => prettyBooleanLiteral(onKey)
            case In(i, typ) => s"${ typ.parsableString() } $i"
            case Die(message, typ) => typ.parsableString()
            case Uniroot(name, _, _, _) => prettyIdentifier(name)
            case CollectDistributedArray(_, _, cname, gname, _) =>
              s"${ prettyIdentifier(cname) } ${ prettyIdentifier(gname) }"
            case MatrixRead(typ, dropCols, dropRows, reader) =>
              (if (typ == reader.fullMatrixType) "None" else typ.parsableString()) + " " +
              prettyBooleanLiteral(dropCols) + " " +
              prettyBooleanLiteral(dropRows) + " " +
              '"' + StringEscapeUtils.escapeString(Serialization.write(reader)(MatrixReader.formats)) + '"'
            case MatrixWrite(_, writer) =>
              '"' + StringEscapeUtils.escapeString(Serialization.write(writer)(MatrixWriter.formats)) + '"'
            case MatrixMultiWrite(_, writer) =>
              '"' + StringEscapeUtils.escapeString(Serialization.write(writer)(MatrixNativeMultiWriter.formats)) + '"'
            case BlockMatrixRead(reader) =>
              '"' + StringEscapeUtils.escapeString(Serialization.write(reader)(BlockMatrixReader.formats)) + '"'
            case BlockMatrixWrite(_, writer) =>
              '"' + StringEscapeUtils.escapeString(Serialization.write(writer)(BlockMatrixWriter.formats)) + '"'
            case BlockMatrixMultiWrite(_, writer) =>
              '"' + StringEscapeUtils.escapeString(Serialization.write(writer)(BlockMatrixWriter.formats)) + '"'
            case BlockMatrixBroadcast(_, inIndexExpr, shape, blockSize) =>
              prettyInts(inIndexExpr) + " " +
              prettyLongs(shape) + " " +
              blockSize.toString + " "
            case BlockMatrixAgg(_, outIndexExpr) => prettyInts(outIndexExpr)
            case BlockMatrixSlice(_, slices) => slices.map(slice => prettyLongs(slice)).mkString("(", " ", ")")
            case ValueToBlockMatrix(_, shape, blockSize) =>
              prettyLongs(shape) + " " +
              blockSize.toString + " "
            case BlockMatrixFilter(_, indicesToKeepPerDim) =>
              indicesToKeepPerDim.map(indices => prettyLongs(indices)).mkString("(", " ", ")")
            case BlockMatrixRandom(seed, gaussian, shape, blockSize) =>
              seed.toString + " " +
              prettyBooleanLiteral(gaussian) + " " +
              prettyLongs(shape) + " " +
              blockSize.toString + " "
            case MatrixRowsHead(_, n) => n.toString
            case MatrixColsHead(_, n) => n.toString
            case MatrixAnnotateRowsTable(_, _, uid, product) =>
              prettyStringLiteral(uid) + " " + prettyBooleanLiteral(product)
            case MatrixAnnotateColsTable(_, _, uid) =>
              prettyStringLiteral(uid)
            case MatrixExplodeRows(_, path) => prettyIdentifiers(path)
            case MatrixExplodeCols(_, path) => prettyIdentifiers(path)
            case MatrixRepartition(_, n, strategy) => s"$n $strategy"
            case MatrixChooseCols(_, oldIndices) => prettyInts(oldIndices)
            case MatrixMapCols(_, _, newKey) => prettyStringsOpt(newKey)
            case MatrixKeyRowsBy(_, keys, isSorted) =>
              prettyIdentifiers(keys) + " " +
                prettyBooleanLiteral(isSorted)
            case TableRead(typ, dropRows, tr) =>
              (if (typ == tr.fullType) "None" else typ.parsableString()) + " " +
                prettyBooleanLiteral(dropRows) + " " +
                '"' + StringEscapeUtils.escapeString(Serialization.write(tr)(TableReader.formats)) + '"'
            case TableWrite(_, writer) =>
              '"' + StringEscapeUtils.escapeString(Serialization.write(writer)(TableWriter.formats)) + '"'
            case TableMultiWrite(_, writer) =>
              '"' + StringEscapeUtils.escapeString(Serialization.write(writer)(WrappedMatrixNativeMultiWriter.formats)) + '"'
            case TableKeyBy(_, keys, isSorted) =>
              prettyIdentifiers(keys) + " " +
                prettyBooleanLiteral(isSorted)
            case TableRange(n, nPartitions) => s"$n $nPartitions"
            case TableRepartition(_, n, strategy) => s"$n $strategy"
            case TableHead(_, n) => n.toString
            case TableJoin(_, _, joinType, joinKey) => s"$joinType $joinKey"
            case TableLeftJoinRightDistinct(_, _, root) => prettyIdentifier(root)
            case TableIntervalJoin(_, _, root, product) =>
              prettyIdentifier(root) + " " + prettyBooleanLiteral(product)
            case TableMultiWayZipJoin(_, dataName, globalName) =>
              s"${ prettyStringLiteral(dataName) } ${ prettyStringLiteral(globalName) }"
            case TableKeyByAndAggregate(_, _, _, nPartitions, bufferSize) =>
              prettyIntOpt(nPartitions) + " " + bufferSize.toString
            case TableExplode(_, path) => prettyStrings(path)
            case TableParallelize(_, nPartitions) =>
                prettyIntOpt(nPartitions)
            case TableOrderBy(_, sortFields) => prettyIdentifiers(sortFields.map(sf =>
              (if (sf.sortOrder == Ascending) "A" else "D") + sf.field))
            case CastMatrixToTable(_, entriesFieldName, colsFieldName) =>
              s"${ prettyStringLiteral(entriesFieldName) } ${ prettyStringLiteral(colsFieldName) }"
            case CastTableToMatrix(_, entriesFieldName, colsFieldName, colKey) =>
              s"${ prettyIdentifier(entriesFieldName) } ${ prettyIdentifier(colsFieldName) } " +
                prettyIdentifiers(colKey)
            case MatrixToMatrixApply(_, function) => prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats))
            case MatrixToTableApply(_, function) => prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats))
            case TableToTableApply(_, function) => prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats))
            case TableToValueApply(_, function) => prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats))
            case MatrixToValueApply(_, function) => prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats))
            case BlockMatrixToTableApply(_, _, function) => prettyStringLiteral(Serialization.write(function)(RelationalFunctions.formats))
            case TableRename(_, rowMap, globalMap) =>
              val rowKV = rowMap.toArray
              val globalKV = globalMap.toArray
              s"${ prettyStrings(rowKV.map(_._1)) } ${ prettyStrings(rowKV.map(_._2)) } " +
                s"${ prettyStrings(globalKV.map(_._1)) } ${ prettyStrings(globalKV.map(_._2)) }"
            case MatrixRename(_, globalMap, colMap, rowMap, entryMap) =>
              val globalKV = globalMap.toArray
              val colKV = colMap.toArray
              val rowKV = rowMap.toArray
              val entryKV = entryMap.toArray
              s"${ prettyStrings(globalKV.map(_._1)) } ${ prettyStrings(globalKV.map(_._2)) } " +
                s"${ prettyStrings(colKV.map(_._1)) } ${ prettyStrings(colKV.map(_._2)) } " +
                s"${ prettyStrings(rowKV.map(_._1)) } ${ prettyStrings(rowKV.map(_._2)) } " +
                s"${ prettyStrings(entryKV.map(_._1)) } ${ prettyStrings(entryKV.map(_._2)) }"
            case TableFilterIntervals(child, intervals, keep) =>
              prettyStringLiteral(Serialization.write(
                JSONAnnotationImpex.exportAnnotation(intervals, TArray(TInterval(child.typ.keyType)))
              )(RelationalSpec.formats)) + " " + prettyBooleanLiteral(keep)
            case MatrixFilterIntervals(child, intervals, keep) =>
              prettyStringLiteral(Serialization.write(
                JSONAnnotationImpex.exportAnnotation(intervals, TArray(TInterval(child.typ.rowKeyStruct)))
              )(RelationalSpec.formats)) + " " + prettyBooleanLiteral(keep)
            case RelationalLetTable(name, _, _) => prettyIdentifier(name)
            case RelationalLetMatrixTable(name, _, _) => prettyIdentifier(name)
            case RelationalLetBlockMatrix(name, _, _) => prettyIdentifier(name)
            case ReadPartition(path, spec, rowType) =>
              s"${ prettyStringLiteral(spec.toString) } ${ rowType.parsableString() }"

            case _ => ""
          }

          if (header.nonEmpty) {
            sb += ' '
            sb.append(header)
          }

          val children = ir.children
          if (children.nonEmpty) {
            sb += '\n'
            children.foreachBetween(c => pretty(c, depth + 2))(sb += '\n')
          }
      }

      sb += ')'
    }

    pretty(ir, 0)

    sb.result()
  }
}
