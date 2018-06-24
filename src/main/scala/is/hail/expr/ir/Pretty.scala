package is.hail.expr.ir

import is.hail.expr.{BaseIR, MatrixRead, TableImport, TableRead}
import is.hail.utils._

object Pretty {
  def prettyStringLiteral(s: String): String =
    "\"" + StringEscapeUtils.escapeString(s) + "\""

  def prettyBooleanLiteral(b: Boolean): String =
    if (b) "True" else "False"

  def prettyClass(x: AnyRef): String =
    x.getClass.getName.split("\\.").last

  def prettyAggSignature(aggSig: AggSignature): String = {
    val sb = new StringBuilder
    sb += '('
    sb.append(prettyClass(aggSig.op))
    sb += ' '
    sb.append(aggSig.inputType.parsableString())
    sb.append(aggSig.constructorArgs.map(_.parsableString()).mkString(" (", " ", ")"))
    sb.append(aggSig.initOpArgs.map(_.map(_.parsableString()).mkString(" (", " ", ")")).getOrElse(" None"))
    sb.append(aggSig.seqOpArgs.map(_.parsableString()).mkString(" (", " ", ")"))
    sb += ')'
    sb.result()
  }

  def apply(ir: BaseIR): String = {
    val sb = new StringBuilder

    def prettySeq(xs: Seq[BaseIR], depth: Int) {
      sb.append(" " * depth)
      sb += '('
      xs.foreachBetween(x => pretty(x, depth + 2))(sb += '\n')
      sb += ')'
    }

    def pretty(ir: BaseIR, depth: Int) {
      sb.append(" " * depth)
      sb += '('

      ir match {
        case _: Apply | _: ApplyIR | _: ApplySpecial =>
          sb.append("Apply")
        case _ =>
          sb.append(prettyClass(ir))
      }

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
        case ApplyAggOp(a, ctorArgs, initOpArgs, aggSig) =>
          sb += ' '
          sb.append(prettyAggSignature(aggSig))
          sb += '\n'
          pretty(a, depth + 2)
          sb += '\n'
          prettySeq(ctorArgs, depth + 2)
          sb += '\n'
          initOpArgs match {
            case Some(initOpArgs) => prettySeq(initOpArgs, depth + 2)
            case None =>
              sb.append(" " * (depth + 2))
              sb.append("None")
          }
        case InitOp(i, args, aggSig) =>
          sb += ' '
          sb.append(prettyAggSignature(aggSig))
          sb += '\n'
          pretty(i, depth + 2)
          sb += '\n'
          prettySeq(args, depth + 2)
        case SeqOp(a, i, aggSig, args) =>
          sb += ' '
          sb.append(prettyAggSignature(aggSig))
          sb += '\n'
          pretty(a, depth + 2)
          sb += '\n'
          pretty(i, depth + 2)
          sb += '\n'
          prettySeq(args, depth + 2)
        case InsertFields(old, fields) =>
          sb += '\n'
          pretty(old, depth + 2)
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
        case TableImport(paths, _, conf) =>
          sb += '\n'
          sb.append(" " * (depth + 2))
          sb.append("(paths\n")
          paths.foreachBetween { p =>
            sb.append(" " * (depth + 4))
            sb.append(prettyStringLiteral(p))
          }(sb += '\n')
          sb += ')'
          sb += '\n'
          sb.append(" " * (depth + 2))
          sb.append("(useCols ")
          conf.useColIndices.foreachBetween(i => sb.append(i))(sb.append(','))
          sb += ')'
          sb += ')'
        case _ =>
          val header = ir match {
            case I32(x) => x.toString
            case I64(x) => x.toString
            case F32(x) => x.toString
            case F64(x) => x.toString
            case Str(x) => prettyStringLiteral(x)
            case Cast(_, typ) => typ.parsableString()
            case NA(typ) => typ.parsableString()
            case Let(name, _, _) => prettyIdentifier(name)
            case Ref(name, typ) => s"${ typ.parsableString() } $name"
            case ApplyBinaryPrimOp(op, _, _) => prettyClass(op)
            case ApplyUnaryPrimOp(op, _) => prettyClass(op)
            case ApplyComparisonOp(op, _, _) => s"(${ prettyClass(op) } ${ op.typ.parsableString() })"
            case GetField(_, name) => prettyIdentifier(name)
            case GetTupleElement(_, idx) => idx.toString
            case MakeArray(_, typ) => typ.parsableString()
            case ArrayMap(_, name, _) => prettyIdentifier(name)
            case ArrayFilter(_, name, _) => prettyIdentifier(name)
            case ArrayFlatMap(_, name, _) => prettyIdentifier(name)
            case ArrayFold(_, _, accumName, valueName, _) => prettyIdentifier(accumName) + " " + prettyIdentifier(valueName)
            case ArrayFor(_, valueName, _) => prettyIdentifier(valueName)
            case ApplyIR(function, _, _) => prettyIdentifier(function)
            case Apply(function, _) => prettyIdentifier(function)
            case ApplySpecial(function, _) => prettyIdentifier(function)
            case SelectFields(_, fields) => fields.map(prettyIdentifier).mkString("(", " ", ")")
            case LowerBoundOnOrderedCollection(_, _, onKey) => prettyBooleanLiteral(onKey)
            case In(i, typ) => s"${ typ.parsableString() } $i"
            case Die(message, typ) => typ.parsableString() + " " + prettyStringLiteral(message)
            case Uniroot(name, _, _, _) => prettyIdentifier(name)
            case MatrixRead(typ, partitionCounts, dropCols, dropRows, requestedType, _) =>
              s"$typ partition_counts=${ partitionCounts.map(_.mkString(",")).getOrElse("None") } ${ if (dropRows) "drop_rows" else "" }${ if (dropCols) "drop_cols" else "" }"
            case TableImport(paths, _, _) =>
              if (paths.length == 1)
                paths.head
              else {
                sb += '\n'
                sb.append(" " * (depth + 2))
                sb.append("(paths\n")
                paths.foreachBetween { p =>
                  sb.append(" " * (depth + 4))
                  sb.append(prettyStringLiteral(p))
                }(sb += '\n')
                sb += ')'

                ""
              }
            case TableRead(path, _, typ, dropRows) =>
              if (dropRows)
                s"${ StringEscapeUtils.escapeString(path) } drop_rows"
              else
                path
            case TableWrite(_, path, overwrite, _) =>
              if (overwrite)
                s"${ StringEscapeUtils.escapeString(path) } overwrite"
              else
                path
            case TableExport(_, path, typesFile, header, exportType) =>
              val args = Array(
                Some(StringEscapeUtils.escapeString(path)),
                Option(typesFile).map(StringEscapeUtils.escapeString(_)),
                if (header) Some("header") else None,
                Some(exportType)
              ).flatten

              sb += '\n'
              args.foreachBetween { a =>
                sb.append(" " * (depth + 2))
                sb.append(a)
              }(sb += '\n')

              ""
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
