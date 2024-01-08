package is.hail.methods

import is.hail.HailContext
import is.hail.annotations.{UnsafeIndexedSeq, UnsafeRow}
import is.hail.backend.ExecuteContext
import is.hail.backend.spark.SparkBackend
import is.hail.expr.TableAnnotationImpex
import is.hail.expr.ir.MatrixValue
import is.hail.expr.ir.functions.MatrixToValueFunction
import is.hail.types.{MatrixType, RTable, TypeWithRequiredness}
import is.hail.types.virtual.{TVoid, Type}
import is.hail.utils._

import java.io.{BufferedOutputStream, OutputStreamWriter}

import org.apache.spark.TaskContext
import org.apache.spark.sql.Row

case class MatrixExportEntriesByCol(
  parallelism: Int,
  path: String,
  bgzip: Boolean,
  headerJsonInFile: Boolean,
  useStringKeyAsFileName: Boolean,
) extends MatrixToValueFunction {
  def typ(childType: MatrixType): Type = TVoid

  def unionRequiredness(childType: RTable, resultType: TypeWithRequiredness): Unit = ()

  def execute(ctx: ExecuteContext, mv: MatrixValue): Any = {
    val fs = ctx.fs

    fs.delete(path, recursive = true) // overwrite by default

    val padding = digitsNeeded(mv.nCols)
    val fileNames: IndexedSeq[String] = if (useStringKeyAsFileName) {
      val ids = mv.stringSampleIds
      if (ids.toSet.size != ids.length) // there are duplicates
        fatal(
          "export_entries_by_col cannot export with 'use_string_key_as_file_name' with duplicate keys"
        )
      ids
    } else
      Array.tabulate(mv.nCols)(i => partFile(padding, i))

    val allColValuesJSON =
      mv.colValues.javaValue.map(TableAnnotationImpex.exportAnnotation(_, mv.typ.colType)).toArray

    val tempFolders = new BoxedArrayBuilder[String]

    info(s"exporting ${mv.nCols} files in batches of $parallelism...")
    val nBatches = (mv.nCols + parallelism - 1) / parallelism
    val resultFiles = (0 until nBatches).flatMap { batch =>
      val startIdx = parallelism * batch
      val nCols = mv.nCols
      val endIdx = math.min(nCols, parallelism * (batch + 1))

      info(s"on batch ${batch + 1} of $nBatches, columns $startIdx to ${endIdx - 1}...")

      val d = digitsNeeded(mv.rvd.getNumPartitions)

      val rvType = mv.rvd.rowPType
      val entriesIdx = MatrixType.getEntriesIndex(rvType)
      val entryArrayType = MatrixType.getEntryArrayType(rvType)
      val entryType = MatrixType.getEntryType(rvType)

      val partFileBase = path + "/tmp/"

      val extension = if (bgzip) ".tsv.bgz" else ".tsv"
      val localHeaderJsonInFile = headerJsonInFile

      val colValuesJSON = HailContext.backend.broadcast(
        (startIdx until endIdx)
          .map(allColValuesJSON)
          .toArray
      )

      val fsBc = fs.broadcast
      val localTempDir = ctx.localTmpdir
      val partFolders = mv.rvd.crdd.cmapPartitionsWithIndex { (i, ctx, it) =>
        val partFolder = partFileBase + partFile(d, i, TaskContext.get())

        val filePaths = Array.tabulate(endIdx - startIdx) { j =>
          val finalPath = partFolder + "/" + j.toString + extension
          val tempPath =
            ExecuteContext.createTmpPathNoCleanup(localTempDir, "EEBC", extension = extension)
          (tempPath, finalPath)
        }

        val fileHandles = filePaths.map { case (tmp, _) =>
          new OutputStreamWriter(new BufferedOutputStream(fsBc.value.create(tmp)), "UTF-8")
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

        it.foreach { ptr =>
          val entriesArray =
            new UnsafeIndexedSeq(entryArrayType, ctx.region, rvType.loadField(ptr, entriesIdx))

          val fullRow = new UnsafeRow(rvType, ctx.region, ptr)

          val rowFieldStrs = (0 until rvType.size)
            .filter(_ != entriesIdx)
            .map { rowFieldIdx =>
              TableAnnotationImpex.exportAnnotation(
                fullRow(rowFieldIdx),
                rvType.types(rowFieldIdx).virtualType,
              )
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
                (0 until entryType.size).foreachBetween(_ => os.write("NA"))(os.write('\t'))
              case r: Row =>
                (0 until entryType.size).foreachBetween { entryFieldIdx =>
                  os.write(TableAnnotationImpex.exportAnnotation(
                    r.get(entryFieldIdx),
                    entryType.types(entryFieldIdx).virtualType,
                  ))
                }(os.write('\t'))
            }

            os.write('\n')
          }
          ctx.region.clear()
        }

        fileHandles.foreach { f =>
          f.flush()
          f.close()
        }
        filePaths.foreach { case (tempFile, destination) =>
          fsBc.value.copy(tempFile, destination, deleteSource = true)
        }

        Iterator(partFolder)
      }.collect()

      val ns = endIdx - startIdx
      val newFiles = mv.sparkContext.parallelize(0 until ns, numSlices = ns)
        .map { sampleIdx =>
          val partFilePath = path + "/" + partFile(digitsNeeded(nCols), sampleIdx, TaskContext.get)
          val fileListEntries =
            partFolders.map(pf => fsBc.value.fileListEntry(pf + s"/$sampleIdx" + extension))
          fsBc.value.copyMergeList(fileListEntries, partFilePath, deleteSource = false)
          partFilePath
        }.collect()

      tempFolders ++= partFolders

      newFiles
    }

    val extension = if (bgzip) ".tsv.bgz" else ".tsv"

    def finalPath(idx: Int): String =
      path + "/" + fileNames(idx) + extension

    resultFiles.zipWithIndex.foreach { case (filePath, i) =>
      fs.copy(filePath, finalPath(i), deleteSource = true)
    }
    fs.delete(path + "/tmp", recursive = true)

    fs.writeTable(
      path + "/index.tsv",
      allColValuesJSON.zipWithIndex.map { case (json, i) =>
        s"${finalPath(i)}\t$json"
      },
    )

    info("Export finished. Cleaning up temporary files...")

    // clean up temporary files
    val temps = tempFolders.result()
    val fsBc = fs.broadcast
    SparkBackend.sparkContext("MatrixExportEntriesByCol.execute").parallelize(
      temps,
      (temps.length / 32).max(1),
    ).foreach(path => fsBc.value.delete(path, recursive = true))

    info("Done cleaning up temporary files.")
  }
}
