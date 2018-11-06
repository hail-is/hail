package is.hail.cxx
import java.io.InputStream

import is.hail.HailContext
import is.hail.annotations.BroadcastRow
import is.hail.expr.ir
import is.hail.expr.types._
import is.hail.expr.types.physical._
import is.hail.io.CodecSpec
import is.hail.rvd.{RVD, RVDContext, RVDSpec}
import is.hail.sparkextras.ContextRDD
import is.hail.table.TableSpec
import is.hail.utils._
import is.hail.variant.{FileFormat, PartitionCountsComponentSpec, RVDComponentSpec, ReferenceGenome}
import org.json4s.jackson.JsonMethods

import scala.reflect.ClassTag

case class TableEmitTriplet(typ: TableType, rvdEmitTriplet: RVDEmitTriplet, globals: BroadcastRow) {

  def mapPartitions(newRowType: TStruct, partSetup: Code)(f: RVDEmitTriplet.ProcessRowF => RVDEmitTriplet.ProcessRowF): TableEmitTriplet =
    TableEmitTriplet(typ.copy(rowType = newRowType), rvdEmitTriplet.mapPartitions(newRowType.physicalType, partSetup)(f), globals)

  def write(tub: TranslationUnitBuilder, path: String, overwrite: Boolean, stageLocally: Boolean, codecSpecJSONStr: String) {
    val hc = HailContext.get

    val codecSpec =
      if (codecSpecJSONStr != null) {
        implicit val formats = RVDSpec.formats
        val codecSpecJSON = JsonMethods.parse(codecSpecJSONStr)
        codecSpecJSON.extract[CodecSpec]
      } else
        CodecSpec.default

    if (overwrite)
      hc.hadoopConf.delete(path, recursive = true)
    else if (hc.hadoopConf.exists(path))
      fatal(s"file already exists: $path")

    hc.hadoopConf.mkDir(path)

    val globalsPath = path + "/globals"
    hc.hadoopConf.mkDir(globalsPath)
    RVDSpec.writeLocal(hc, globalsPath, typ.globalType, codecSpec, Array(globals.value))

    val partitionCounts = RVDEmitTriplet.write(rvdEmitTriplet, tub, path + "/rows", stageLocally, codecSpec)

    val referencesPath = path + "/references"
    hc.hadoopConf.mkDir(referencesPath)
    ReferenceGenome.exportReferences(hc, referencesPath, typ.rowType)
    ReferenceGenome.exportReferences(hc, referencesPath, typ.globalType)

    val spec = TableSpec(
      FileFormat.version.rep,
      hc.version,
      "references",
      typ,
      Map("globals" -> RVDComponentSpec("globals"),
        "rows" -> RVDComponentSpec("rows"),
        "partition_counts" -> PartitionCountsComponentSpec(partitionCounts)))
    spec.write(hc, path)

    hc.hadoopConf.writeTextFile(path + "/_SUCCESS")(out => ())


  }
}

object TableEmit {
  def apply(tub: TranslationUnitBuilder, x: ir.TableIR): TableEmitTriplet = {
    val emitter = new TableEmitter(tub)
    emitter.emit(x)
  }
}

class TableEmitter(tub: TranslationUnitBuilder) { outer =>
  type E = ir.Env[TableEmitTriplet]

  def emit(x: ir.TableIR): TableEmitTriplet = emit(x, ir.Env.empty[TableEmitTriplet])

  def emit(x: ir.TableIR, env: E): TableEmitTriplet = {

    def emit(x: ir.TableIR, env: E = env): TableEmitTriplet = this.emit(x, env)

    val typ = x.typ
    x match {
      case ir.TableRead(path, spec, _, dropRows) =>
        val hc = HailContext.get
        val globals = spec.globalsComponent.readLocal(hc, path, typ.globalType)(0)
        val rvd = if (dropRows)
          RVDEmitTriplet.empty[InputStream](typ.rvdType)
        else {
          val rvd = spec.rowsComponent.cxxEmitRead(hc, path, typ.rowType, tub)
//          if (rvd.typ.key startsWith typ.key)
//            rvd
//          else {
//            log.info("Sorting a table after read. Rewrite the table to prevent this in the future.")
//            rvd.changeKey(typ.key)
//          }
          rvd
        }
        new TableEmitTriplet(typ, rvd, BroadcastRow(globals, typ.globalType, hc.sc))

      case ir.TableMapRows(child, newRow) =>

        val fName = genSym("map_row")
        val mapF = FunctionBuilder(fName, Array("NativeStatus*" -> "st", "Region*" -> "region", "char *" -> "row"), "char *")
        val substEnv = ir.Env.empty[ir.IR].bind("row", ir.In(1, child.typ.rowType))

        val et = Emit(mapF, 1, ir.Subst(newRow, substEnv))

        mapF +=
          s"""
             |${ et.setup }
             |if (${ et.m }) {
             |  ${ mapF.nativeError(1011, "\"mapped row can't be missing!\"") }
             |} else {
             |  return ${ et.v };
             |}
           """.stripMargin

        tub += mapF.result()

        val prev = emit(child)
        prev.mapPartitions(newRow.typ.asInstanceOf[TStruct], "") { processRowF =>
          { row: Variable =>
            val newRow = Variable("mapped_row", "char *", s"$fName(${ prev.rvdEmitTriplet.st }, ${ prev.rvdEmitTriplet.region }, $row)")
            s"""
               |${ newRow.define }
               |${ processRowF(newRow) }
               """.stripMargin
          }
        }

      case ir.TableFilter(child, cond) =>
        val fName = genSym("filter_cond")
        val condF = FunctionBuilder(fName, Array("NativeStatus*" -> "st", "Region*" -> "region", "char *" -> "row"), "bool")
        val substEnv = ir.Env.empty[ir.IR].bind("row", ir.In(1, child.typ.rowType))
        val et = Emit(condF, 1, ir.Subst(cond, substEnv))

        condF +=
          s"""
             |${ et.setup }
             |return (${ et.m }) || (${ et.v });
           """.stripMargin
        tub += condF.result()

        val prev = emit(child)
        prev.mapPartitions(prev.typ.rowType, "") { processRowF =>
          { row: Variable =>
            s"""
               |if ($fName(${ prev.rvdEmitTriplet.st }, ${ prev.rvdEmitTriplet.region }, $row)) {
               |  ${ processRowF(row) }
               |}
               """.stripMargin
          }
        }

      case _ =>
        throw new CXXUnsupportedOperation(ir.Pretty(x))
    }
  }
}
