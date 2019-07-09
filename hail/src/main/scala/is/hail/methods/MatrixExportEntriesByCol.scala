package is.hail.methods

import java.io.OutputStreamWriter

import is.hail.HailContext
import is.hail.annotations.{UnsafeIndexedSeq, UnsafeRow}
import is.hail.expr.TableAnnotationImpex
import is.hail.expr.ir.{ExecuteContext, MatrixValue}
import is.hail.expr.ir.functions.MatrixToValueFunction
import is.hail.expr.types.MatrixType
import is.hail.expr.types.virtual.{TVoid, Type}
import is.hail.utils._
import org.apache.spark.TaskContext
import org.apache.spark.sql.Row

case class MatrixExportEntriesByCol(parallelism: Int, path: String, bgzip: Boolean, headerJsonInFile: Boolean) extends MatrixToValueFunction {
  def typ(childType: MatrixType): Type = TVoid

  def execute(ctx: ExecuteContext, mv: MatrixValue): Any = {

    val fs = HailContext.sFS

    fs.delete(path, recursive = true) // overwrite by default

    val allColValuesJSON = mv.colValues.javaValue.map(TableAnnotationImpex.exportAnnotation(_, mv.typ.colType)).toArray

    info(s"exporting ${ mv.nCols } files in batches of $parallelism...")
    val nBatches = (mv.nCols + parallelism - 1) / parallelism
    val resultFiles = (0 until nBatches).flatMap { batch =>
      val startIdx = parallelism * batch
      val nCols = mv.nCols
      val endIdx = math.min(nCols, parallelism * (batch + 1))

      info(s"on batch ${ batch + 1 } of ${ nBatches }, columns $startIdx to ${ endIdx - 1 }...")

      val d = digitsNeeded(mv.rvd.getNumPartitions)

      val rvType = mv.rvd.rowPType
      val entriesIdx = MatrixType.getEntriesIndex(rvType)
      val entryArrayType = MatrixType.getEntryArrayType(rvType)
      val entryType = MatrixType.getEntryType(rvType)

      val partFileBase = path + "/tmp/"

      val bcFS = HailContext.bcFS

      val extension = if (bgzip) ".tsv.bgz" else ".tsv"
      val localHeaderJsonInFile = headerJsonInFile

      val colValuesJSON = HailContext.backend.broadcast(
        (startIdx until endIdx)
          .map(allColValuesJSON)
          .toArray)

      val partFolders = mv.rvd.crdd.mapPartitionsWithIndex { (i, it) =>

        val partFolder = partFileBase + partFile(d, i, TaskContext.get())

        val fileHandles = Array.tabulate(endIdx - startIdx) { j =>
          new OutputStreamWriter(bcFS.value.unsafeWriter(partFolder + "/" + j.toString + extension), "UTF-8")
        }

        if (i == 0) {
          // write headers
          val header = (
            rvType.fieldNames.filter(_ != MatrixType.entriesIdentifier) ++ entryType.fieldNames
            ).mkString("\t")

          fileHandles.zipWithIndex.foreach { case (f, jj) =>
            if (localHeaderJsonInFile) {
              f.write('#')
              f.write(colValuesJSON.value(jj))
              f.write('\n')
            }
            f.write(header)
            f.write('\n')
          }
        }

        it.foreach { rv =>

          val entriesArray = new UnsafeIndexedSeq(entryArrayType, rv.region, rvType.loadField(rv, entriesIdx))

          val fullRow = new UnsafeRow(rvType, rv)

          val rowFieldStrs = (0 until rvType.size)
            .filter(_ != entriesIdx)
            .map { rowFieldIdx =>
              TableAnnotationImpex.exportAnnotation(fullRow(rowFieldIdx), rvType.types(rowFieldIdx).virtualType)
            }.toArray

          fileHandles.indices.foreach { fileIdx =>
            val entryIdx = fileIdx + startIdx
            val os = fileHandles(fileIdx)

            rowFieldStrs.foreach { s =>
              os.write(s)
              os.write('\t')
            }

            entriesArray(entryIdx) match {
              case null =>
                (0 until entryType.size).foreachBetween { _ =>
                  os.write("NA")
                }(os.write('\t'))
              case r: Row =>
                (0 until entryType.size).foreachBetween { entryFieldIdx =>
                  os.write(TableAnnotationImpex.exportAnnotation(r.get(entryFieldIdx), entryType.types(entryFieldIdx).virtualType))
                }(os.write('\t'))
            }

            os.write('\n')
          }
        }

        fileHandles.foreach(_.close())
        Iterator(partFolder)
      }.collect()

      val ns = endIdx - startIdx
      val newFiles = mv.sparkContext.parallelize(0 until ns, numSlices = ns)
        .map { sampleIdx =>
          val partFilePath = path + "/" + partFile(digitsNeeded(nCols), sampleIdx, TaskContext.get)
          val fileStatuses = partFolders.map(pf => bcFS.value.fileStatus(pf + s"/$sampleIdx" + extension))
          bcFS.value.copyMergeList(fileStatuses, partFilePath, deleteSource = true)
          partFilePath
        }.collect()

      newFiles
    }

    val padding = digitsNeeded(mv.nCols)
    val extension = if (bgzip) ".tsv.bgz" else ".tsv"

    def finalPath(idx: Int): String = path + "/" + partFile(padding, idx) + extension

    resultFiles.zipWithIndex.foreach { case (filePath, i) =>
      fs.copy(filePath, finalPath(i), deleteSource = true)
    }
    fs.delete(path + "/tmp", recursive = true)

    fs.writeTable(path + "/index.tsv", allColValuesJSON.zipWithIndex.map { case (json, i) =>
      s"${ finalPath(i) }\t$json"
    })

    info("export finished!")
  }
}
