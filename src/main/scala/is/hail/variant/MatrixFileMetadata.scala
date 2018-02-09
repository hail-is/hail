package is.hail.variant

import is.hail.annotations.Annotation
import is.hail.expr.MatrixLocalValue
import is.hail.expr.types._

object MatrixFileMetadata {
  def apply(matrixType: MatrixType, localValue: MatrixLocalValue): MatrixFileMetadata = MatrixFileMetadata(matrixType, localValue, None)

  def apply(sampleAnnotations: IndexedSeq[Annotation],
    globalAnnotation: Annotation = Annotation.empty,
    saSignature: TStruct = TStruct("s" -> TString()),
    colKey: IndexedSeq[String] = Array("s"),
    vaSignature: TStruct = TStruct("v" -> TVariant(GenomeReference.defaultReference)),
    rowKey: IndexedSeq[String] = Array("v"),
    rowPartitionKey: IndexedSeq[String] = Array("v"),
    globalSignature: TStruct = TStruct.empty(),
    // FIXME require
    genotypeSignature: TStruct = Genotype.htsGenotypeType): MatrixFileMetadata = {
    MatrixFileMetadata(
      MatrixType.fromParts(globalSignature, colKey, saSignature, rowPartitionKey, rowKey, vaSignature, genotypeSignature),
      MatrixLocalValue(globalAnnotation, sampleAnnotations))
  }
}

case class MatrixFileMetadata(
  matrixType: MatrixType,
  localValue: MatrixLocalValue,
  partitionCounts: Option[Array[Long]]) {
}
